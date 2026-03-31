import os
import sys
import numpy as np
import torch
import gradio as gr
import spaces  # noqa: F401
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("HF_REPO_ID", "").strip() == "":
    os.environ["HF_REPO_ID"] = "openbmb/VoxCPM2"

import voxcpm


class VoxCPMDemo:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Running on device: {self.device}", file=sys.stderr)

        # ASR model for prompt text recognition
        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )

        # TTS model (lazy init)
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self.default_local_model_dir = "/Users/xinliu/Downloads/VoxCPM2-0.5B-newaudiovae-6hz-0316"

    # ---------- Model helpers ----------
    def _resolve_model_dir(self) -> str:
        """
        Resolve model directory:
        1) Use local checkpoint directory if exists
        2) If HF_REPO_ID env is set, download into models/{repo}
        3) Fallback to 'models'
        """
        if os.path.isdir(self.default_local_model_dir):
            return self.default_local_model_dir

        repo_id = os.environ.get("HF_REPO_ID", "").strip()
        if len(repo_id) > 0:
            target_dir = os.path.join("models", repo_id.replace("/", "__"))
            if not os.path.isdir(target_dir):
                try:
                    from huggingface_hub import snapshot_download  # type: ignore

                    os.makedirs(target_dir, exist_ok=True)
                    print(f"Downloading model from HF repo '{repo_id}' to '{target_dir}' ...", file=sys.stderr)
                    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
                except Exception as e:
                    print(f"Warning: HF download failed: {e}. Falling back to 'data'.", file=sys.stderr)
                    return "models"
            return target_dir
        return "models"

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        print("Model not loaded, initializing...", file=sys.stderr)
        model_dir = self._resolve_model_dir()
        print(f"Using model dir: {model_dir}", file=sys.stderr)
        self.voxcpm_model = voxcpm.VoxCPM(voxcpm_model_path=model_dir, optimize=False)
        print("Model loaded successfully.", file=sys.stderr)
        return self.voxcpm_model

    # ---------- Functional endpoints ----------
    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        text = res[0]["text"].split("|>")[-1]
        return text

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        cfg_value_input: float = 2.0,
        inference_timesteps_input: int = 10,
        do_normalize: bool = True,
        denoise: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate speech from text using VoxCPM.
        - If reference_wav provided: Prompt isolation mode (voice cloning)
        - If no reference_wav: Voice design mode (use control_instruction to describe voice)

        Returns (sample_rate, waveform_numpy)
        """
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        # 处理 control instruction
        control = (control_instruction or "").strip()
        if control:
            final_text = f"({control}){text}"
        else:
            final_text = text

        reference_wav_path = reference_wav_path_input if reference_wav_path_input else None

        # 判断模式
        if reference_wav_path:
            print(f"[Prompt Isolation Mode] reference_wav: {reference_wav_path}", file=sys.stderr)
        else:
            print(f"[Voice Design Mode] control: {control[:50] if control else 'None'}...", file=sys.stderr)

        print(f"Generating audio for text: '{final_text[:80]}...'", file=sys.stderr)
        wav = current_model.generate(
            text=final_text,
            reference_wav_path=reference_wav_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=int(inference_timesteps_input),
            normalize=do_normalize,
            denoise=denoise,
        )
        return (current_model.tts_model.sample_rate, wav)


# ---------- UI Builders ----------

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

CSS = """
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
/* Bold accordion labels */
#acc_quick > .label-wrap,
#acc_tips > .label-wrap,
#acc_quick > .label-wrap > span,
#acc_tips > .label-wrap > span,
#acc_quick summary,
#acc_tips summary {
    font-weight: 600 !important;
    font-size: 1.1em !important;
}
/* Bold labels for specific checkboxes */
#chk_denoise label,
#chk_denoise span,
#chk_normalize label,
#chk_normalize span {
    font-weight: 600;
}
"""


def create_demo_interface(demo: VoxCPMDemo):
    """Build the Gradio UI for VoxCPM demo."""
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container"><img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo"></div>',
            padding=True,
        )

        # Quick Start
        with gr.Accordion("📋 Quick Start Guide ｜快速入门", open=False, elem_id="acc_quick"):
            gr.Markdown("""
            ### How to Use ｜使用说明
            1. **(Optional) Provide a Voice Prompt** - Upload or record an audio clip to provide the desired voice characteristics for synthesis.  
               **（可选）提供参考声音** - 上传或录制一段音频，为声音合成提供音色、语调和情感等个性化特征
            2. **(Optional) Enter prompt text** - If you provided a voice prompt, enter the corresponding transcript here (auto-recognition available).  
               **（可选项）输入参考文本** - 如果提供了参考语音，请输入其对应的文本内容（支持自动识别）。
            3. **Enter target text** - Type the text you want the model to speak.  
               **输入目标文本** - 输入您希望模型朗读的文字内容。
            4. **Generate Speech** - Click the "Generate" button to create your audio.  
               **生成语音** - 点击"生成"按钮，即可为您创造出音频。
            """)

        # Pro Tips
        with gr.Accordion("💡 Pro Tips ｜使用建议", open=False, elem_id="acc_tips"):
            gr.Markdown("""
            ### Prompt Speech Enhancement｜参考语音降噪
            - **Enable** to remove background noise for a clean voice, with an external ZipEnhancer component. However, this will limit the audio sampling rate to 16kHz, restricting the cloning quality ceiling.  
              **启用**：通过 ZipEnhancer 组件消除背景噪音，但会将音频采样率限制在16kHz，限制克隆上限。
            - **Disable** to preserve the original audio's all information, including background atmosphere, and support audio cloning up to 44.1kHz sampling rate.  
              **禁用**：保留原始音频的全部信息，包括背景环境声，最高支持44.1kHz的音频复刻。

            ### Text Normalization｜文本正则化
            - **Enable** to process general text with an external WeTextProcessing component.  
              **启用**：使用 WeTextProcessing 组件，可支持常见文本的正则化处理。
            - **Disable** to use VoxCPM's native text understanding ability. For example, it supports phonemes input (For Chinese, phonemes are converted using pinyin, {ni3}{hao3}; For English, phonemes are converted using CMUDict, {HH AH0 L OW1}), try it!  
              **禁用**：将使用 VoxCPM 内置的文本理解能力。如，支持音素输入（如中文转拼音：{ni3}{hao3}；英文转CMUDict：{HH AH0 L OW1}）和公式符号合成，尝试一下！

            ### CFG Value｜CFG 值
            - **Lower CFG** if the voice prompt sounds strained or expressive, or instability occurs with long text input.  
              **调低**：如果提示语音听起来不自然或过于夸张，或者长文本输入出现稳定性问题。
            - **Higher CFG** for better adherence to the prompt speech style or input text, or instability occurs with too short text input.
              **调高**：为更好地贴合提示音频的风格或输入文本， 或者极短文本输入出现稳定性问题。

            ### Inference Timesteps｜推理时间步
            - **Lower** for faster synthesis speed.  
              **调低**：合成速度更快。
            - **Higher** for better synthesis quality.  
              **调高**：合成质量更佳。
            """)

        # Main controls
        with gr.Row():
            with gr.Column():
                # 1. Reference Audio
                # gr.Markdown("### 🎤 Reference Audio (Optional)")
                # gr.Markdown("*提供参考音频进行音色克隆；不提供则使用 Voice Design 模式*")
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Reference Audio (Optional)",
                )
                DoDenoisePromptAudio = gr.Checkbox(
                    value=False,
                    label="Reference Audio Enhancement",
                    elem_id="chk_denoise",
                    info="Use ZipEnhancer to denoise the reference audio",
                )

                # 2. Control Instruction
                # gr.Markdown("### 🎛️ Control Instruction (Optional)")
                # gr.Markdown("*描述声音风格、情感等，格式：`(instruction) text`*")
                control_instruction = gr.Textbox(
                    value="",
                    label="Control Instruction",
                    placeholder="*描述声音风格、情感等，格式：`(instruction) text`，例如：年轻女性，温柔甜美 / 悲伤地说 / an excited young man*",
                    lines=2,
                )

                # 3. Target Text
                # gr.Markdown("### 📝 Target Text")
                text = gr.Textbox(
                    value="VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.",
                    label="Target Text",
                    lines=3,
                )
                DoNormalizeText = gr.Checkbox(
                    value=False,
                    label="Text Normalization",
                    elem_id="chk_normalize",
                    info="Use wetext library to normalize the input text",
                )

                run_btn = gr.Button("🔊 Generate Speech", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### ⚙️ Generation Settings")
                cfg_value = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="CFG Value (Guidance Scale)",
                    info="Higher = more adherence to prompt; Lower = more creativity",
                )
                inference_timesteps = gr.Slider(
                    minimum=4,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Inference Timesteps",
                    info="Higher = better quality but slower",
                )

                gr.Markdown("### 🔈 Output")
                audio_output = gr.Audio(label="Generated Audio")

                gr.Markdown("""
                ---
                **模式说明 / Mode Info:**
                - **有 Reference Audio** → Prompt 隔离模式（音色克隆）
                - **无 Reference Audio** → Voice Design 模式（用 Control Instruction 描述声音）
                
                **Control Instruction 示例：**
                - `年轻女性，温柔甜美`
                - `悲伤地说`
                - `an excited young man`
                """)

        # Wiring
        run_btn.click(
            fn=demo.generate_tts_audio,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                cfg_value,
                inference_timesteps,
                DoNormalizeText,
                DoDenoisePromptAudio,
            ],
            outputs=[audio_output],
            show_progress=True,
            api_name="generate",
        )

    return interface


def run_demo(server_name: str = "0.0.0.0", server_port: int = 7869, show_error: bool = True):
    demo = VoxCPMDemo()
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    run_demo()
