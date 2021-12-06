# Copyright 2021 TUNiB inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import suppress
from typing import List, Union

from torch import nn

from parallelformers.policies.base import Policy


class AutoPolicy:
    """Class for finds automatically appropriate policies for the current model"""

    def __init__(self):
        self.builtin_policies = {}

        with suppress(Exception):
            from transformers.models.gpt_neo.modeling_gpt_neo import (
                GPTNeoPreTrainedModel,
            )

            from parallelformers.policies.gpt_neo import GPTNeoPolicy

            self.builtin_policies[GPTNeoPreTrainedModel] = [
                GPTNeoPolicy,
            ]

        with suppress(Exception):
            from transformers.models.bert.modeling_bert import (
                BertPreTrainedModel,
            )

            from parallelformers.policies.bert import BertPolicy

            self.builtin_policies[BertPreTrainedModel] = [
                BertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.bart.modeling_bart import (
                BartPretrainedModel,
            )

            from parallelformers.policies.bart import (
                BartDecoderPolicy,
                BartEncoderPolicy,
            )

            self.builtin_policies[BartPretrainedModel] = [
                BartEncoderPolicy,
                BartDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.blenderbot.modeling_blenderbot import (
                BlenderbotPreTrainedModel,
            )

            from parallelformers.policies.blenderbot import (
                BlenderbotDecoderPolicy,
                BlenderbotEncoderPolicy,
            )

            self.builtin_policies[BlenderbotPreTrainedModel] = [
                BlenderbotEncoderPolicy,
                BlenderbotDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.deberta.modeling_deberta import (
                DebertaPreTrainedModel,
            )

            from parallelformers.policies.deberta import DebertaPolicy

            self.builtin_policies[DebertaPreTrainedModel] = [
                DebertaPolicy,
            ]

        with suppress(Exception):
            from transformers.models.transfo_xl.modeling_transfo_xl import (
                TransfoXLPreTrainedModel,
            )

            from parallelformers.policies.transfo_xl import TransfoXLPolicy

            self.builtin_policies[TransfoXLPreTrainedModel] = [
                TransfoXLPolicy,
            ]

        with suppress(Exception):
            from transformers.models.roberta.modeling_roberta import (
                RobertaPreTrainedModel,
            )

            from parallelformers.policies.roberta import RobertaPolicy

            self.builtin_policies[RobertaPreTrainedModel] = [
                RobertaPolicy,
            ]

        with suppress(Exception):
            from transformers.models.albert.modeling_albert import (
                AlbertPreTrainedModel,
            )

            from parallelformers.policies.albert import AlbertPolicy

            self.builtin_policies[AlbertPreTrainedModel] = [
                AlbertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.gpt2.modeling_gpt2 import (
                GPT2PreTrainedModel,
            )

            from parallelformers.policies.gpt2 import GPT2Policy

            self.builtin_policies[GPT2PreTrainedModel] = [
                GPT2Policy,
            ]

        with suppress(Exception):
            from transformers.models.ctrl.modeling_ctrl import (
                CTRLPreTrainedModel,
            )

            from parallelformers.policies.ctrl import CTRLPolicy

            self.builtin_policies[CTRLPreTrainedModel] = [
                CTRLPolicy,
            ]

        with suppress(Exception):
            from transformers.models.deberta_v2.modeling_deberta_v2 import (
                DebertaV2PreTrainedModel,
            )

            from parallelformers.policies.deberta_v2 import DebertaV2Policy

            self.builtin_policies[DebertaV2PreTrainedModel] = [
                DebertaV2Policy,
            ]

        with suppress(Exception):
            from transformers.models.openai.modeling_openai import (
                OpenAIGPTPreTrainedModel,
            )

            from parallelformers.policies.openai import OpenAIGPTPolicy

            self.builtin_policies[OpenAIGPTPreTrainedModel] = [
                OpenAIGPTPolicy,
            ]

        with suppress(Exception):
            from transformers.models.electra.modeling_electra import (
                ElectraPreTrainedModel,
            )

            from parallelformers.policies.electra import ElectraPolicy

            self.builtin_policies[ElectraPreTrainedModel] = [
                ElectraPolicy,
            ]

        with suppress(Exception):
            from transformers.models.blenderbot_small.modeling_blenderbot_small import (
                BlenderbotSmallPreTrainedModel,
            )

            from parallelformers.policies.blenderbot_small import (
                BlenderbotSmallDecoderPolicy,
                BlenderbotSmallEncoderPolicy,
            )

            self.builtin_policies[BlenderbotSmallPreTrainedModel] = [
                BlenderbotSmallEncoderPolicy,
                BlenderbotSmallDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.distilbert.modeling_distilbert import (
                DistilBertPreTrainedModel,
            )

            from parallelformers.policies.distil_bert import DistilBertPolicy

            self.builtin_policies[DistilBertPreTrainedModel] = [
                DistilBertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.convbert.modeling_convbert import (
                ConvBertPreTrainedModel,
            )

            from parallelformers.policies.convbert import ConvBertPolicy

            self.builtin_policies[ConvBertPreTrainedModel] = [
                ConvBertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.bert_generation.modeling_bert_generation import (
                BertGenerationPreTrainedModel,
            )

            from parallelformers.policies.bert import BertPolicy

            self.builtin_policies[BertGenerationPreTrainedModel] = [
                BertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.big_bird.modeling_big_bird import (
                BigBirdPreTrainedModel,
            )

            from parallelformers.policies.bigbird import BigBirdPolicy

            self.builtin_policies[BigBirdPreTrainedModel] = [
                BigBirdPolicy,
            ]

        with suppress(Exception):
            from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
                BigBirdPegasusPreTrainedModel,
            )

            from parallelformers.policies.bigbird_pegasus import (
                BigBirdPegasusDecoderPolicy,
                BigBirdPegasusEncoderPolicy,
            )

            self.builtin_policies[BigBirdPegasusPreTrainedModel] = [
                BigBirdPegasusEncoderPolicy,
                BigBirdPegasusDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.vit.modeling_vit import ViTPreTrainedModel

            from parallelformers.policies.vit import ViTPolicy

            self.builtin_policies[ViTPreTrainedModel] = [
                ViTPolicy,
            ]

        with suppress(Exception):
            from transformers.models.deit.modeling_deit import (
                DeiTPreTrainedModel,
            )

            from parallelformers.policies.deit import DeiTPolicy

            self.builtin_policies[DeiTPreTrainedModel] = [DeiTPolicy]

        with suppress(Exception):
            from transformers.models.mbart.modeling_mbart import (
                MBartPreTrainedModel,
            )

            from parallelformers.policies.mbart import (
                MBartDecoderPolicy,
                MBartEncoderPolicy,
            )

            self.builtin_policies[MBartPreTrainedModel] = [
                MBartEncoderPolicy,
                MBartDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.t5.modeling_t5 import T5PreTrainedModel

            from parallelformers.policies.t5 import T5Policy

            self.builtin_policies[T5PreTrainedModel] = [
                T5Policy,
            ]

        with suppress(Exception):
            from transformers.models.pegasus.modeling_pegasus import (
                PegasusPreTrainedModel,
            )

            from parallelformers.policies.pegasus import (
                PegasusDecoderPolicy,
                PegasusEncoderPolicy,
            )

            self.builtin_policies[PegasusPreTrainedModel] = [
                PegasusEncoderPolicy,
                PegasusDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.fsmt.modeling_fsmt import (
                PretrainedFSMTModel,
            )

            from parallelformers.policies.fsmt import (
                FSMTDecoderPolicy,
                FSMTEncoderPolicy,
            )

            self.builtin_policies[PretrainedFSMTModel] = [
                FSMTEncoderPolicy,
                FSMTDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.xlm.modeling_xlm import XLMPreTrainedModel

            from parallelformers.policies.xlm import (
                XLMAttentionPolicy,
                XLMMLPPolicy,
            )

            self.builtin_policies[XLMPreTrainedModel] = [
                XLMAttentionPolicy,
                XLMMLPPolicy,
            ]

        with suppress(Exception):
            from transformers.models.m2m_100.modeling_m2m_100 import (
                M2M100PreTrainedModel,
            )

            from parallelformers.policies.m2m_100 import (
                M2M100DecoderPolicy,
                M2M100EncoderPolicy,
            )

            self.builtin_policies[M2M100PreTrainedModel] = [
                M2M100EncoderPolicy,
                M2M100DecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.marian.modeling_marian import (
                MarianPreTrainedModel,
            )

            from parallelformers.policies.marian import (
                MarianDecoderPolicy,
                MarianEncoderPolicy,
            )

            self.builtin_policies[MarianPreTrainedModel] = [
                MarianEncoderPolicy,
                MarianDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.mobilebert.modeling_mobilebert import (
                MobileBertPreTrainedModel,
            )

            from parallelformers.policies.mobilebert import MobileBertPolicy

            self.builtin_policies[MobileBertPreTrainedModel] = [
                MobileBertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.mpnet.modeling_mpnet import (
                MPNetPreTrainedModel,
            )

            from parallelformers.policies.mpnet import (
                MPNetEncoderPolicy,
                MPNetLayerPolicy,
            )

            self.builtin_policies[MPNetPreTrainedModel] = [
                MPNetEncoderPolicy,
                MPNetLayerPolicy,
            ]

        with suppress(Exception):
            from transformers.models.luke.modeling_luke import (
                LukePreTrainedModel,
            )

            from parallelformers.policies.luke import LukePolicy

            self.builtin_policies[LukePreTrainedModel] = [
                LukePolicy,
            ]

        with suppress(Exception):
            from transformers.models.dpr.modeling_dpr import (
                DPRPretrainedContextEncoder,
                DPRPretrainedQuestionEncoder,
                DPRPretrainedReader,
            )

            self.builtin_policies[DPRPretrainedReader] = [
                BertPolicy,
            ]

            self.builtin_policies[DPRPretrainedQuestionEncoder] = [
                BertPolicy,
            ]

            self.builtin_policies[DPRPretrainedContextEncoder] = [
                BertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.lxmert.modeling_lxmert import (
                LxmertPreTrainedModel,
            )

            from parallelformers.policies.lxmert import LxmertPolicy

            self.builtin_policies[LxmertPreTrainedModel] = [
                LxmertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.hubert.modeling_hubert import (
                HubertPreTrainedModel,
            )

            from parallelformers.policies.hubert import HubertPolicy

            self.builtin_policies[HubertPreTrainedModel] = [
                HubertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2PreTrainedModel,
            )

            from parallelformers.policies.wav2vec import Wav2VecPolicy

            self.builtin_policies[Wav2Vec2PreTrainedModel] = [
                Wav2VecPolicy,
            ]

        with suppress(Exception):
            from transformers.models.xlnet.modeling_xlnet import (
                XLNetPreTrainedModel,
            )

            from parallelformers.policies.xlnet import XLNetPolicy

            self.builtin_policies[XLNetPreTrainedModel] = [
                XLNetPolicy,
            ]

        with suppress(Exception):
            from transformers.models.retribert.modeling_retribert import (
                RetriBertPreTrainedModel,
            )

            self.builtin_policies[RetriBertPreTrainedModel] = [
                BertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.clip.modeling_clip import (
                CLIPPreTrainedModel,
            )

            from parallelformers.policies.clip import (
                CLIPLayerPolicy,
                CLIPTextPolicy,
                CLIPVisionPolicy,
            )

            self.builtin_policies[CLIPPreTrainedModel] = [
                CLIPLayerPolicy,
                CLIPTextPolicy,
                CLIPVisionPolicy,
            ]

        with suppress(Exception):
            from transformers.models.detr.modeling_detr import (
                DetrPreTrainedModel,
            )

            from parallelformers.policies.detr import (
                DetrDecoderPolicy,
                DetrEncoderPolicy,
            )

            self.builtin_policies[DetrPreTrainedModel] = [
                DetrEncoderPolicy,
                DetrDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.reformer.modeling_reformer import (
                ReformerPreTrainedModel,
            )

            from parallelformers.policies.reformer import ReformerPolicy

            self.builtin_policies[ReformerPreTrainedModel] = [
                ReformerPolicy,
            ]

        with suppress(Exception):
            from transformers.models.longformer.modeling_longformer import (
                LongformerPreTrainedModel,
            )

            from parallelformers.policies.longformer import LongformerPolicy

            self.builtin_policies[LongformerPreTrainedModel] = [
                LongformerPolicy,
            ]

        with suppress(Exception):
            from transformers.models.roformer.modeling_roformer import (
                RoFormerPreTrainedModel,
            )

            from parallelformers.policies.roformer import RoformerPolicy

            self.builtin_policies[RoFormerPreTrainedModel] = [
                RoformerPolicy,
            ]

        with suppress(Exception):
            from transformers.models.ibert.modeling_ibert import (
                IBertPreTrainedModel,
            )

            from parallelformers.policies.ibert import IBertPolicy

            self.builtin_policies[IBertPreTrainedModel] = [
                IBertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.tapas.modeling_tapas import (
                TapasPreTrainedModel,
            )

            from parallelformers.policies.tapas import TapasPolicy

            self.builtin_policies[TapasPreTrainedModel] = [
                TapasPolicy,
            ]

        with suppress(Exception):
            from transformers.models.funnel.modeling_funnel import (
                FunnelPreTrainedModel,
            )

            from parallelformers.policies.funnel import FunnelPolicy

            self.builtin_policies[FunnelPreTrainedModel] = [
                FunnelPolicy,
            ]

        with suppress(Exception):
            from transformers.models.layoutlm.modeling_layoutlm import (
                LayoutLMPreTrainedModel,
            )

            from parallelformers.policies.layoutlm import LayoutLMPolicy

            self.builtin_policies[LayoutLMPreTrainedModel] = [
                LayoutLMPolicy,
            ]

        with suppress(Exception):
            from transformers.models.led.modeling_led import LEDPreTrainedModel

            from parallelformers.policies.led import (
                LEDDecoderPolicy,
                LEDEncoderPolicy,
            )

            self.builtin_policies[LEDPreTrainedModel] = [
                LEDEncoderPolicy,
                LEDDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.prophetnet.modeling_prophetnet import (
                ProphetNetPreTrainedModel,
            )

            from parallelformers.policies.prophetnet import (
                ProphetNetDecoderPolicy,
                ProphetNetEncoderPolicy,
            )

            self.builtin_policies[ProphetNetPreTrainedModel] = [
                ProphetNetEncoderPolicy,
                ProphetNetDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.visual_bert.modeling_visual_bert import (
                VisualBertPreTrainedModel,
            )

            from parallelformers.policies.visual_bert import VisualBertPolicy

            self.builtin_policies[VisualBertPreTrainedModel] = [
                VisualBertPolicy,
            ]

        with suppress(Exception):
            from transformers.models.speech_to_text.modeling_speech_to_text import (
                Speech2TextPreTrainedModel,
            )

            from parallelformers.policies.speech_to_text import (
                Speech2TextDecoderPolicy,
                Speech2TextEncoderPolicy,
            )

            self.builtin_policies[Speech2TextPreTrainedModel] = [
                Speech2TextEncoderPolicy,
                Speech2TextDecoderPolicy,
            ]

        with suppress(Exception):
            from transformers.models.gptj.modeling_gptj import (
                GPTJPreTrainedModel,
            )

            from parallelformers.policies.gptj import GPTJPolicy

            self.builtin_policies[GPTJPreTrainedModel] = [
                GPTJPolicy,
            ]

        with suppress(Exception):
            from transformers.models.megatron_bert import (
                MegatronBertPreTrainedModel,
            )

            from parallelformers.policies.megtron_bert import (
                MegatronBertPolicy,
            )

            self.builtin_policies[MegatronBertPreTrainedModel] = [
                MegatronBertPolicy,
            ]

    def get_policy(self, model: nn.Module) -> Union[List[Policy], None]:
        """
        Find appropriate policies for the current model

        Args:
            model (nn.Module): model to parallelize

        Returns:
            Union[List[Policy], None]: appropriate policies or none
        """
        for k, v in self.available().items():
            if isinstance(model, k):
                return v

        return None

    def available(self):
        """Dictionary of available models and policies"""
        return self.builtin_policies
