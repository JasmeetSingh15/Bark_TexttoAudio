from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor(
    text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
    return_tensors="pt", voice_preset=voice_preset
)

speech_values = model.generate(**inputs, do_sample=True)

import scipy

sampling_rate = model.config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
