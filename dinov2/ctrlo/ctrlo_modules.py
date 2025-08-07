from functools import partial

import torch
from torch import nn

from dinov2.ctrlo import typing
from dinov2.ctrlo.losses import ReconstructionLoss, DiagonalContrastiveLoss
from dinov2.ctrlo.modules.conditioning import LangConditioning
from dinov2.ctrlo.modules.decoder_conditioning import EncodeLangConditioning
from dinov2.ctrlo.modules.decoding import PatchDecoder
from dinov2.ctrlo.modules.heads import LangEmbeddingHead, PointEmbeddingHead, AttentionAggregationHead, \
    SlotProjectorHead
from dinov2.ctrlo.modules.mapping import MLPMapping
from dinov2.ctrlo.modules.perceptual_grouping import SlotAttentionGrouping
from dinov2.ctrlo.nns.convenience import build_mlp, build_two_layer_mlp
from dinov2.ctrlo.nns.positional_embedding import DummyPositionEmbed
from dinov2.ctrlo.nns.wrappers import Sequential
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.utils.utils import type_or_shape


class CTRLOWrapper(nn.Module):
    def __init__(
        self,
        num_slots: int,
        feature_dim: int,
        slot_dim: int,
        num_patches: int,
        lang_dim: int = 4096,
        embedding_dim: int = 512,
    ):
        object_dim = slot_dim
        super().__init__()
        self.conditioning = LangConditioning(
            n_slots=num_slots,
            object_dim=object_dim,
            lang_dim=lang_dim,
            dual_conditioning=False,
        )

        self.lang_embedding = LangEmbeddingHead(
            embedding_dim=embedding_dim,
            lang_dim=lang_dim,
        )

        self.point_embedding = PointEmbeddingHead(
            embedding_dim=embedding_dim,
        )

        self.dec_conditioning = EncodeLangConditioning(
            dim=slot_dim,
            lang_dim=lang_dim,
        )

        self.mapping = MLPMapping(
            dim=feature_dim,
        )


        self.perceptual_grouping = SlotAttentionGrouping(
            feature_dim=object_dim,
            object_dim=slot_dim,
            use_projection_bias=False,
            positional_embedding=Sequential(
                DummyPositionEmbed(),
                build_two_layer_mlp(
                    input_dim=feature_dim,
                    output_dim=object_dim,
                    hidden_dim=feature_dim*2,
                    initial_layer_norm=True
                )
            ),
            ff_mlp=build_two_layer_mlp(
                input_dim=object_dim,
                output_dim=object_dim,
                hidden_dim=object_dim*4,
                initial_layer_norm=True,
                residual=True,
            )
        )

        self.attn_aggregation = AttentionAggregationHead(
            dim=feature_dim,
        )

        self.object_decoder = PatchDecoder(
            decoder=partial(
                build_mlp,
                features=[2048, 2048, 2048],
            ),
            object_dim=slot_dim,
            output_dim=feature_dim,
            num_patches=num_patches,
            conditioned=True,
        )

        self.projector_slots = SlotProjectorHead(
            dim=feature_dim,
            embedding_dim=embedding_dim,
        )

        self.loss_fn = CTRLOLosses()


    def forward(self, inputs_dict: dict, feature_extractor: DinoVisionTransformer):
        from pprint import pprint
        # pprint(type_or_shape(inputs_dict))

        # inputs_dict = {
        #     k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
        #     for (k,v) in inputs_dict.items()
        # }


        H = feature_extractor.forward_features(inputs_dict["image"])["x_norm_patchtokens"]

        fe_out = typing.FeatureExtractorOutput(
            features=H.float(),
            positions=torch.empty((H.shape[0], 0, 0))
        )

        cond_out = self.conditioning.forward(
            name_embedding=inputs_dict["name_embedding"],
            batch_size=inputs_dict["batch_size"],
            mask=inputs_dict["contrastive_loss_mask"],
        )

        lang_out = self.lang_embedding.forward(
            name_embedding=inputs_dict["name_embedding"],
        )

        point_out = self.point_embedding.forward(
            point_embedding=inputs_dict["bbox_centroids"],
        )

        dec_out = self.dec_conditioning.forward(
            language=inputs_dict["name_embedding"],
            mask=inputs_dict["contrastive_loss_mask"],
        )

        map_out = self.mapping.forward(
            x=fe_out,
        )


        pg_out = self.perceptual_grouping.forward(
            feature=map_out,
            conditioning=cond_out,
        )

        attn_agg_out = self.attn_aggregation.forward(
            attn=pg_out.feature_attributions,
            x=map_out.features,
        )

        dec_out = self.object_decoder.forward(
            object_features=pg_out.objects,
            image=inputs_dict["image"],
            condition_info=dec_out,
        )

        proj_out = self.projector_slots.forward(
            slots=attn_agg_out,
        )


        ctrlo_loss = self.loss_fn.forward(
            ft_features=fe_out.features,
            dec_reconstruction=dec_out.reconstruction,
            proj_slots=proj_out,
            lang_embedding=lang_out,
            point_embedding=point_out,
            contrastive_mask=inputs_dict["contrastive_loss_mask"],
        )

        return ctrlo_loss

class CTRLOLosses(nn.Module):
    def __init__(
            self,
            # reconstruction_loss_weight: float = 1.0,
            # contrastive_loss_lang_weight: float = 0.2,
            # contrastive_loss_point_weight: float = 0.2,
    ):
        super().__init__()
        self.mse = ReconstructionLoss(
            loss_type="mse",
            # weight=1, #reconstruction_loss_weight,
        )
        self.contrastive_loss_lang = DiagonalContrastiveLoss(
            temp=0.1,
            batch_contrastive=True,
            # weight=contrastive_loss_lang_weight,
        )

        self.contrastive_loss_point = DiagonalContrastiveLoss(
            temp=0.1,
            batch_contrastive=True,
            # weight=contrastive_loss_point_weight,
        )

    def forward(
            self,
            ft_features,
            dec_reconstruction,
            proj_slots,
            lang_embedding,
            point_embedding,
            contrastive_mask
    ):

        mse_loss = self.mse.forward(
            input=dec_reconstruction,
            target=ft_features,
        )

        lang_loss = self.contrastive_loss_lang.forward(
            x1=proj_slots,
            x2=lang_embedding,
            contrastive_loss_mask=contrastive_mask,
        )

        point_loss = self.contrastive_loss_point.forward(
            x1=proj_slots,
            x2=point_embedding,
            contrastive_loss_mask=contrastive_mask,
        )

        return dict(
            mse_loss=mse_loss,
            lang_loss=lang_loss,
            point_loss=point_loss,
        )
        # print(type_or_shape([
        #     ft_features,
        #     dec_reconstruction,
        #     proj_slots,
        #     lang_embedding,
        #     point_embedding,
        #     contrastive_mask,
        #     mse_loss, lang_loss, point_loss,
        # ]))

        # return mse_loss + lang_loss + point_loss




