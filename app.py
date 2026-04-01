import os
import sys
import logging
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM2"

import voxcpm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**Usage Instructions:**\n\n"
    "🎨 **Voice Design** — Create a voice from scratch  \n"
    "No reference audio needed. Simply describe the desired gender, tone, and emotion "
    "in Control Instruction, and VoxCPM will generate a unique voice for you.\n\n"
    "🎛️ **Controllable Voice Cloning** — Clone with style control  \n"
    "Upload reference audio and use Control Instruction to guide speed, emotion, style, and more.\n\n"
    "🎙️ **Hi-Fi Cloning** — Maximum voice similarity  \n"
    "For the best cloning quality, enable and provide the reference audio transcript "
    "to reproduce the original voice as closely as possible."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**Voice Description Examples:**  \n"
    "You can describe it like this:  \n"
    "【Example 1: Melancholic/Tsundere Female】  \n"
    'Control Instruction: "A young beautiful girl with a sweet voice, '
    'tsundere tone, slow speaking pace, and a touch of sadness."  \n'
    'Target Text: "I never asked you to stay... It\'s not like I care or anything. '
    'But... why does it still hurt so much now that you\'re gone?"  \n\n'
    "【Example 2: Lazy/Casual Male】  \n"
    'Control Instruction: "Lazy and drawling male voice, nasal, '
    'very relaxed and casual."  \n'
    'Target Text: "Dude, did you see that set? The waves out there are totally gnarly today, bro. '
    "Just catching barrels all morning. It's like, totally righteous, you know what I mean?\""
)

_USAGE_INSTRUCTIONS_ZH = (
    "**使用说明：**\n\n"
    "🎨 **Voice Design — 声音定制**  \n"
    "无需上传参考音频，只需在 Control Instruction 中描述你想要的性别、音色和情绪，"
    "VoxCPM 即可凭空为你生成专属音色。\n\n"
    "🎛️ **Controllable Voice Cloning — 可控音色克隆**  \n"
    "支持上传参考音频，并可以给instruction文本来指导控制语速、情绪、风格等表现。\n\n"
    "🎙️ **Hi-Fi Cloning — 高保真克隆**  \n"
    "追求最佳克隆效果，启用并上传参考音频文本来最大程度克隆原始音色。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**声音描述示例：**  \n"
    "你可以这样输入（中英文均可）：  \n"
    "【示例1：深宫太后】  \n"
    '`Control Instruction`: `"中老年女性，声音低沉阴冷，语速慢而有力，'
    '每个字都像是深思熟虑后说出，带有深不可测的城府和威胁感。"`  \n'
    '`Target Text`: `"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"`  \n\n'
    "【示例2：暴躁男声】  \n"
    '`Control Instruction`: `"暴躁的中年男声，语速较快，充满无奈和愤怒"`  \n'
    '`Target Text`: `"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"`\n\n'
    "💡 **方言生成特别说明：**  \n"
    '当前版本若要生成纯正的方言，请务必在"Target Text"中直接输入方言专属的词汇和表达，'
    "并配合方言的音色描述。  \n\n"
    "【示例一：广东话】  \n"
    '`Control Instruction`: `"广东话，中年男性，语气平淡"`  \n'
    "✅ 正确的 `Target Text`（使用粤语表达）："
    '`"伙計，唔該一個A餐，凍奶茶少甜！"`  \n'
    "❌ 错误的 `Target Text`（使用普通话）："
    '`"伙计，麻烦来一个A餐，冻奶茶少甜！"`  \n\n'
    "【示例二：河南话】  \n"
    '`Control Instruction`: `"河南话，接地气的大叔"`  \n'
    "✅ 正确的 `Target Text`（使用河南话表达）："
    '`"恁这是弄啥嘞？晌午吃啥饭？"`  \n'
    "❌ 错误的 `Target Text`（使用普通话）："
    '`"你这是在干什么呢？中午吃什么饭？"`  \n\n'
    "🤖 **实用小技巧：不知道怎么写地道的方言？**  \n"
    "您可以先在 豆包、DeepSeek、Kimi 等 AI 助手中输入普通话，"
    "让它们帮你翻译成方言文本，然后再复制粘贴到 `Target Text` 中直接使用！  \n\n"
    "📢 **研发小贴士：**  \n"
    '我们正在努力优化 AI！后续版本将支持"输入普通话文本，一键生成方言口音"的功能，敬请期待！'
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "Reference Audio (optional — for cloning)",
        "show_prompt_text_label": "Enable Prompt Text (improves voice similarity)",
        "show_prompt_text_info": "Uses the ASR transcript of reference audio for higher cloning fidelity. Control Instruction will be disabled.",
        "prompt_text_label": "Prompt Text (auto-filled by ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here...",
        "control_label": "Control Instruction (optional, only support English and Chinese)",
        "control_placeholder": "e.g. 年轻女性，温柔甜美 / sadly / an excited young man",
        "target_text_label": "Target Text",
        "generate_btn": "Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Denoise reference audio with ZipEnhancer",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize input text with wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher = stronger prompt adherence; lower = more variation",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "参考音频（可选 - 用于克隆）",
        "show_prompt_text_label": "启用 Prompt Text（提升音色还原度）",
        "show_prompt_text_info": "使用参考音频的文本内容提升克隆相似度，开启后 Control Instruction 将被禁用",
        "prompt_text_label": "Prompt Text（ASR 自动填充，可编辑）",
        "prompt_text_placeholder": "参考音频的文本内容将自动识别到这里...",
        "control_label": "Control Instruction（可选，仅支持中文和英文）",
        "control_placeholder": "如：年轻女性，温柔甜美 / sadly / an excited young man",
        "target_text_label": "Target Text（要合成的文本）",
        "generate_btn": "开始生成",
        "generated_audio_label": "生成音频",
        "advanced_settings_title": "高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "使用 ZipEnhancer 对参考音频进行降噪",
        "normalize_label": "文本规范化",
        "normalize_info": "使用 wetext 对输入文本进行规范化处理",
        "cfg_label": "CFG Value（引导强度）",
        "cfg_info": "数值越高，越贴合提示要求；数值越低，变化空间越大",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM is an innovative end-to-end TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------

class VoxCPMDemo:
    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on device: {self.device}")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.explicit_model_dir = model_dir

    def _resolve_model_dir(self) -> str:
        if self.explicit_model_dir and os.path.isdir(self.explicit_model_dir):
            return self.explicit_model_dir
        env_model_dir = os.environ.get("VOXCPM_MODEL_DIR", "").strip()
        if env_model_dir and os.path.isdir(env_model_dir):
            return env_model_dir
        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download
                    os.makedirs(target_dir, exist_ok=True)
                    logger.info(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...")
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    logger.warning(f"HF download failed: {e}. Falling back to 'models'.")
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info("Model not loaded, initializing...")
        model_dir = self._resolve_model_dir()
        logger.info(f"Using model dir: {model_dir}")
        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir, optimize=True)
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=10,
            normalize=do_normalize,
            denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        sr, wav_np = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
        )
        return (sr, wav_np)

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        run_btn.click(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface

def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_dir: Optional[str] = None,
):
    demo = VoxCPMDemo(model_dir=model_dir)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None, help="Path to VoxCPM2 checkpoint directory")
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    args = parser.parse_args()
    run_demo(model_dir=args.model_dir, server_port=args.port)
