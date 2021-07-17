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

from typing import List, Union

from torch import nn
from transformers.models.albert.modeling_albert import AlbertPreTrainedModel
from transformers.models.bart.modeling_bart import BartPretrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert_generation.modeling_bert_generation import (
    BertGenerationPreTrainedModel,
)
from transformers.models.big_bird.modeling_big_bird import (
    BigBirdPreTrainedModel,
)
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
    BigBirdPegasusPreTrainedModel,
)
from transformers.models.blenderbot.modeling_blenderbot import (
    BlenderbotPreTrainedModel,
)
from transformers.models.blenderbot_small.modeling_blenderbot_small import (
    BlenderbotSmallPreTrainedModel,
)
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel
from transformers.models.convbert.modeling_convbert import (
    ConvBertPreTrainedModel,
)
from transformers.models.ctrl.modeling_ctrl import CTRLPreTrainedModel
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
)
from transformers.models.deit.modeling_deit import DeiTPreTrainedModel
from transformers.models.detr.modeling_detr import DetrPreTrainedModel
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertPreTrainedModel,
)
from transformers.models.dpr.modeling_dpr import (
    DPRPretrainedContextEncoder,
    DPRPretrainedQuestionEncoder,
    DPRPretrainedReader,
)
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.models.fsmt.modeling_fsmt import PretrainedFSMTModel
from transformers.models.funnel.modeling_funnel import FunnelPreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoPreTrainedModel
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel
from transformers.models.ibert.modeling_ibert import IBertPreTrainedModel
from transformers.models.layoutlm.modeling_layoutlm import (
    LayoutLMPreTrainedModel,
)
from transformers.models.led.modeling_led import LEDPreTrainedModel
from transformers.models.longformer.modeling_longformer import (
    LongformerPreTrainedModel,
)
from transformers.models.luke.modeling_luke import LukePreTrainedModel
from transformers.models.lxmert.modeling_lxmert import LxmertPreTrainedModel
from transformers.models.m2m_100.modeling_m2m_100 import M2M100PreTrainedModel
from transformers.models.marian.modeling_marian import MarianPreTrainedModel
from transformers.models.mbart.modeling_mbart import MBartPreTrainedModel
from transformers.models.mobilebert.modeling_mobilebert import (
    MobileBertPreTrainedModel,
)
from transformers.models.mpnet.modeling_mpnet import MPNetPreTrainedModel
from transformers.models.openai.modeling_openai import OpenAIGPTPreTrainedModel
from transformers.models.pegasus.modeling_pegasus import PegasusPreTrainedModel
from transformers.models.prophetnet.modeling_prophetnet import (
    ProphetNetPreTrainedModel,
)
from transformers.models.reformer.modeling_reformer import (
    ReformerPreTrainedModel,
)
from transformers.models.retribert.modeling_retribert import (
    RetriBertPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.roformer.modeling_roformer import (
    RoFormerPreTrainedModel,
)
from transformers.models.speech_to_text.modeling_speech_to_text import (
    Speech2TextPreTrainedModel,
)
from transformers.models.t5.modeling_t5 import T5PreTrainedModel
from transformers.models.tapas.modeling_tapas import TapasPreTrainedModel
from transformers.models.transfo_xl.modeling_transfo_xl import (
    TransfoXLPreTrainedModel,
)
from transformers.models.visual_bert.modeling_visual_bert import (
    VisualBertPreTrainedModel,
)
from transformers.models.vit.modeling_vit import ViTPreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
)
from transformers.models.xlm.modeling_xlm import XLMPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetPreTrainedModel

from parallelformers.policies.albert import AlbertPolicy
from parallelformers.policies.bart import BartDecoderPolicy, BartEncoderPolicy
from parallelformers.policies.base import Policy
from parallelformers.policies.bert import BertPolicy
from parallelformers.policies.bigbird import BigBirdPolicy
from parallelformers.policies.bigbird_pegasus import (
    BigBirdPegasusDecoderPolicy,
    BigBirdPegasusEncoderPolicy,
)
from parallelformers.policies.blenderbot import (
    BlenderbotDecoderPolicy,
    BlenderbotEncoderPolicy,
)
from parallelformers.policies.blenderbot_small import (
    BlenderbotSmallDecoderPolicy,
    BlenderbotSmallEncoderPolicy,
)
from parallelformers.policies.clip import (
    CLIPLayerPolicy,
    CLIPTextPolicy,
    CLIPVisionPolicy,
)
from parallelformers.policies.convbert import ConvBertPolicy
from parallelformers.policies.ctrl import CTRLPolicy
from parallelformers.policies.deberta import DebertaPolicy
from parallelformers.policies.deberta_v2 import DebertaV2Policy
from parallelformers.policies.deit import DeiTPolicy
from parallelformers.policies.detr import DetrDecoderPolicy, DetrEncoderPolicy
from parallelformers.policies.distil_bert import DistilBertPolicy
from parallelformers.policies.electra import ElectraPolicy
from parallelformers.policies.fsmt import FSMTDecoderPolicy, FSMTEncoderPolicy
from parallelformers.policies.funnel import FunnelPolicy
from parallelformers.policies.gpt2 import GPT2Policy
from parallelformers.policies.gpt_neo import GPTNeoPolicy
from parallelformers.policies.hubert import HubertPolicy
from parallelformers.policies.ibert import IBertPolicy
from parallelformers.policies.layoutlm import LayoutLMPolicy
from parallelformers.policies.led import LEDDecoderPolicy, LEDEncoderPolicy
from parallelformers.policies.longformer import LongformerPolicy
from parallelformers.policies.luke import LukePolicy
from parallelformers.policies.lxmert import LxmertPolicy
from parallelformers.policies.m2m_100 import (
    M2M100DecoderPolicy,
    M2M100EncoderPolicy,
)
from parallelformers.policies.marian import (
    MarianDecoderPolicy,
    MarianEncoderPolicy,
)
from parallelformers.policies.mbart import (
    MBartDecoderPolicy,
    MBartEncoderPolicy,
)
from parallelformers.policies.mobilebert import MobileBertPolicy
from parallelformers.policies.mpnet import MPNetEncoderPolicy, MPNetLayerPolicy
from parallelformers.policies.openai import OpenAIGPTPolicy
from parallelformers.policies.pegasus import (
    PegasusDecoderPolicy,
    PegasusEncoderPolicy,
)
from parallelformers.policies.prophetnet import (
    ProphetNetDecoderPolicy,
    ProphetNetEncoderPolicy,
)
from parallelformers.policies.reformer import ReformerPolicy
from parallelformers.policies.roberta import RobertaPolicy
from parallelformers.policies.roformer import RoformerPolicy
from parallelformers.policies.speech_to_text import (
    Speech2TextDecoderPolicy,
    Speech2TextEncoderPolicy,
)
from parallelformers.policies.t5 import T5Policy
from parallelformers.policies.tapas import TapasPolicy
from parallelformers.policies.transfo_xl import TransfoXLPolicy
from parallelformers.policies.visual_bert import VisualBertPolicy
from parallelformers.policies.vit import ViTPolicy
from parallelformers.policies.wav2vec import Wav2VecPolicy
from parallelformers.policies.xlm import XLMAttentionPolicy, XLMMLPPolicy
from parallelformers.policies.xlnet import XLNetPolicy


class AutoPolicy:
    """Class for finds automatically appropriate policies for the current model"""

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

    @staticmethod
    def available():
        """Dictionary of available models and policies"""
        return {
            GPTNeoPreTrainedModel: [GPTNeoPolicy],
            BertPreTrainedModel: [BertPolicy],
            BartPretrainedModel: [
                BartEncoderPolicy,
                BartDecoderPolicy,
            ],
            BlenderbotPreTrainedModel: [
                BlenderbotEncoderPolicy,
                BlenderbotDecoderPolicy,
            ],
            DebertaPreTrainedModel: [DebertaPolicy],
            TransfoXLPreTrainedModel: [TransfoXLPolicy],
            RobertaPreTrainedModel: [RobertaPolicy],
            AlbertPreTrainedModel: [AlbertPolicy],
            GPT2PreTrainedModel: [GPT2Policy],
            CTRLPreTrainedModel: [CTRLPolicy],
            DebertaV2PreTrainedModel: [DebertaV2Policy],
            OpenAIGPTPreTrainedModel: [OpenAIGPTPolicy],
            ElectraPreTrainedModel: [ElectraPolicy],
            BlenderbotSmallPreTrainedModel: [
                BlenderbotSmallEncoderPolicy,
                BlenderbotSmallDecoderPolicy,
            ],
            DistilBertPreTrainedModel: [DistilBertPolicy],
            ConvBertPreTrainedModel: [ConvBertPolicy],
            BertGenerationPreTrainedModel: [BertPolicy],
            BigBirdPreTrainedModel: [BigBirdPolicy],
            BigBirdPegasusPreTrainedModel: [
                BigBirdPegasusEncoderPolicy,
                BigBirdPegasusDecoderPolicy,
            ],
            ViTPreTrainedModel: [ViTPolicy],
            DeiTPreTrainedModel: [DeiTPolicy],
            MBartPreTrainedModel: [
                MBartEncoderPolicy,
                MBartDecoderPolicy,
            ],
            T5PreTrainedModel: [T5Policy],
            PegasusPreTrainedModel: [
                PegasusEncoderPolicy,
                PegasusDecoderPolicy,
            ],
            PretrainedFSMTModel: [
                FSMTEncoderPolicy,
                FSMTDecoderPolicy,
            ],
            XLMPreTrainedModel: [
                XLMAttentionPolicy,
                XLMMLPPolicy,
            ],
            M2M100PreTrainedModel: [
                M2M100EncoderPolicy,
                M2M100DecoderPolicy,
            ],
            MarianPreTrainedModel: [
                MarianEncoderPolicy,
                MarianDecoderPolicy,
            ],
            MobileBertPreTrainedModel: [MobileBertPolicy],
            MPNetPreTrainedModel: [
                MPNetLayerPolicy,
                MPNetEncoderPolicy,
            ],
            LukePreTrainedModel: [LukePolicy],
            DPRPretrainedContextEncoder: [BertPolicy],
            DPRPretrainedQuestionEncoder: [BertPolicy],
            DPRPretrainedReader: [BertPolicy],
            LxmertPreTrainedModel: [LxmertPolicy],
            HubertPreTrainedModel: [HubertPolicy],
            Wav2Vec2PreTrainedModel: [Wav2VecPolicy],
            XLNetPreTrainedModel: [XLNetPolicy],
            RetriBertPreTrainedModel: [BertPolicy],
            CLIPPreTrainedModel: [
                CLIPLayerPolicy,
                CLIPTextPolicy,
                CLIPVisionPolicy,
            ],
            DetrPreTrainedModel: [
                DetrEncoderPolicy,
                DetrDecoderPolicy,
            ],
            ReformerPreTrainedModel: [ReformerPolicy],
            LongformerPreTrainedModel: [LongformerPolicy],
            RoFormerPreTrainedModel: [RoformerPolicy],
            IBertPreTrainedModel: [IBertPolicy],
            TapasPreTrainedModel: [TapasPolicy],
            FunnelPreTrainedModel: [FunnelPolicy],
            LayoutLMPreTrainedModel: [LayoutLMPolicy],
            LEDPreTrainedModel: [
                LEDEncoderPolicy,
                LEDDecoderPolicy,
            ],
            ProphetNetPreTrainedModel: [
                ProphetNetEncoderPolicy,
                ProphetNetDecoderPolicy,
            ],
            VisualBertPreTrainedModel: [VisualBertPolicy],
            Speech2TextPreTrainedModel: [
                Speech2TextEncoderPolicy,
                Speech2TextDecoderPolicy,
            ],
        }
