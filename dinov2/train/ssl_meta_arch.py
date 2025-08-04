# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from einops import einsum
from torch import nn

from dinov2.ctrlo.ctrlo_modules import CTRLOWrapper
from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk
import torch.nn.functional as F

try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        # self.register_contrastive_loss_weight = cfg.student.register_contrastive_loss_weight
        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        self.ctrlo_model = CTRLOWrapper(
            num_slots=cfg.ctrlo.num_slots,
            slot_dim=cfg.ctrlo.slot_dim,
            feature_dim=embed_dim,
            num_patches=(cfg.crops.global_crops_size // cfg.student.patch_size) ** 2
        )

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images_ctrlo_and_dino_input, teacher_temp):

        images_dino = images_ctrlo_and_dino_input["image_dino"]
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images_dino["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images_dino["collated_local_crops"].cuda(non_blocking=True)

        masks = images_dino["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images_dino["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images_dino["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images_dino["upperbound"]
        masks_weight = images_dino["masks_weight"].cuda(non_blocking=True)
        # register_global_prompts = images["collated_global_labels"].cuda(non_blocking=True)
        # register_global_negatives = images["collated_global_negatives"].cuda(non_blocking=True)
        # register_local_prompts = images["collated_local_labels"].cuda(non_blocking=True)


        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops

            # assert False, (register_global_prompts.shape, register_global_negatives.shape)
            # teacher_register_input = torch.cat([register_global_prompts.unsqueeze(1), register_global_negatives], dim=1)
            # print("TEACHER")
            teacher_register_input = None
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True, register_prompts=teacher_register_input, return_attention=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            if teacher_backbone_output_dict["last_attn"] is not None:

                tri = teacher_register_input.shape[1] if teacher_register_input is not None else 0
                reg_end = 1 + (self.cfg.student.num_register_tokens * tri)
                register_patch_mean_attn = teacher_backbone_output_dict["last_attn"][:, :, 1:reg_end, reg_end:].mean(dim=1)
                # mean along the head dim

                patches = teacher_backbone_output_dict["x_norm_patchtokens"]

                teacher_patch_reg_representations = einsum(
                    register_patch_mean_attn,
                    patches,
                    "b r p, b p e -> b r e"
                )
                                # from einops import einsum
                # assert False, (register_patch_mean_attn.shape, patches.shape, patch_reg_representations.shape)

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_reg_representations

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered, teacher_patch_reg_representations = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop

        # print("STUDENT GLOBAL")
        register_global_prompts = None
        student_global_backbone_output_dict = self.student.backbone(
            global_crops, masks=masks, is_training=True, register_prompts=register_global_prompts, return_attention=True
        )

        # assert False, local_
        # crops.shape
        # print("STUDENT LOCAL")
        register_local_prompts = None
        student_local_backbone_output_dict = self.student.backbone(
            local_crops, masks=None, is_training=True, register_prompts=register_local_prompts, return_attention=True
        )

        # student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
        #     [global_crops, local_crops], masks=[masks, None], is_training=True, register_prompts=[register_global_prompts, register_local_prompts]
        # )

        # assert False, {
        #     k: (v.shape if isinstance(v, torch.Tensor) else type(v))
        #     for (k,v) in student_global_backbone_output_dict.items()
        # }
        # assert False, teacher_backbone_output_dict.keys()

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if self.cfg.ctrlo.loss_weight > 0:

            non_dino_input = {
                k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k,v
                in images_ctrlo_and_dino_input.items()
                if k != "image_dino"
            }
            ctrlo_loss = self.ctrlo_model.forward(non_dino_input, feature_extractor=self.student.backbone)
            loss_dict["ctrlo_loss"] = ctrlo_loss
            loss_accumulator += ctrlo_loss * self.cfg.ctrlo.loss_weight

        # TODO remove the below probably
        # if self.register_contrastive_loss_weight > 0:
        #     assert False, teacher_patch_reg_representations.shape
        #     # B, 1+r_neg, E = teacher_patch_reg_representations.shape
        #     reg_end = 1 + (self.cfg.student.num_register_tokens)
        #     student_register_mean_attn = student_local_backbone_output_dict["last_attn"][:, :, 1:reg_end, reg_end:].mean(dim=1)
        #     # assert False, student_register_mean_attn.shape
        #     # register_patch_mean_attn = teacher_backbone_output_dict["last_attn"][:, :, 1:reg_end, reg_end:].mean(dim=1)
        #     # mean along the head dim
        #
        #     student_local_patches = student_local_backbone_output_dict["x_norm_patchtokens"]
        #
        #     # assert False, [t.shape for t in [student_local_patches, student_register_mean_attn,  teacher_patch_reg_representations]]
        #
        #     student_local_patch_reg_representations = einsum(
        #         student_register_mean_attn,
        #         student_local_patches,
        #         "b r p, b p e -> b r e"
        #     )
        #
        #     teacher_patch_reg_representations_norm = F.normalize(teacher_patch_reg_representations, eps=1e-8, p=2, dim=-1).chunk(n_global_crops)
        #     student_local_patch_reg_representations_norm = F.normalize(student_local_patch_reg_representations, eps=1e-8, p=2, dim=-1).chunk(n_local_crops)
        #
        #     contr_loss_agg = 0
        #     for tr in teacher_patch_reg_representations_norm:
        #         for sr in student_local_patch_reg_representations_norm:
        #             cont_map = einsum(sr, tr, "b sr e, b tr e -> b sr tr")
        #             b, srn, trn = cont_map.shape
        #             assert srn == 1, cont_map.shape
        #             cont_map = cont_map.squeeze()
        #             assert cont_map.shape == (b, trn)
        #             labels = torch.zeros(len(cont_map), dtype=torch.long).to(cont_map.device)
        #             cont_loss = F.cross_entropy(cont_map, labels)
        #             contr_loss_agg += cont_loss
        #
        #             # assert False, [tr.shape, sr.shape, cont_map.shape]
        #     loss_dict["semantic_register_contrastive_loss"] = contr_loss_agg
        #     loss_accumulator += contr_loss_agg * self.register_contrastive_loss_weight

            # print(register_global_prompts)
            #
            # print(register_local_prompts)
            #
            # assert False, [[t_.shape for t_ in t] for t in [teacher_patch_reg_representations_norm, student_local_patch_reg_representations_norm]]
            #
            # assert False, [t.shape for t in [student_local_patch_reg_representations_norm, teacher_patch_reg_representations_norm]]
            # inter_sample_contrastive = einsum(
            #     student_local_patch_reg_representations_norm,
            #     teacher_patch_reg_representations_norm,
            #     "sb sr e"
            # )

            # assert False, [t.shape for t in [student_local_patches, student_register_mean_attn,  student_local_patch_reg_representations, teacher_patch_reg_representations]]

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            for attr in {
                "_unshard_stream",
                "_post_backward_stream",
                "_pre_unshard_stream",
                "_all_reduce_stream",
                "_default_stream"
            }:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)
            # self.student.dino_head._streams = (
            #     self.teacher.dino_head._streams
            # ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
