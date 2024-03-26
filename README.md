# Bark_TexttoAudio
Suno Bark github URL: https://github.com/suno-ai/bark
Suno Bark HuggingFace URL: https://huggingface.co/suno/bark
Coqui AI URL: https://github.com/coqui-ai/TTS
Language Supports in Bark URL: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
Bark Repo in HuggingFace URL: https://huggingface.co/suno/bark/tree/main

Try closing all the application before starting with the steps given below for the ease of reproducibility. 

Steps:
0. Create a specific folder for the project.
1. Install Python. Link: https://www.python.org/downloads/. After installation make sure you have the PATH populated with the bin and scripts folder of python.
2. Install Git Bash. Link: https://git-scm.com/downloads. Make sure you have location of git bash bin added to the PATH environment variable.
3. Install Visual Studio Code. Link: https://code.visualstudio.com/download.
4. Install Anaconda. Link: https://docs.anaconda.com/free/anaconda/install/index.html. While installing make sure to click the visual studio code option. This would automatically integrate Visual Studio Code and Anaconda.
5. Launch Anaconda Navigator.
6. Find Visual Studio Code as an application displayed on the Anaconda navigator. Click on launch.
7. Open the folder which was created as part of the step 0. From inside of Visual Studio Code open the terminal.
8. In the terminal type "python --version" to get the exact version of python you have installed.
9. In the terminal type "conda create -n bark python=<version_found_in_previous_step>"
10. In the terminal enable the newly created conda environment "conda activate bark".
11. Now we need to install two packages first "pip install git+https://github.com/suno-ai/bark.git" and second "pip install git+https://github.com/huggingface/transformers.git".
12. Create a new file and name it as "audio_generator.py".
13. Copy the code from part two of https://github.com/suno-ai/bark#-transformers-usage. Same code is available in "audio_generation_old.py" or you can use the code in "audio_generator.py".
14. From the terminal type "python ./audio_generator.py". If you just have CPU it would take some time for this to run.
15. This would take some time depending on the hardware configuration you have. After its complete it would generate a file called bark_generation.wav.
16. Play "bark_generation.wav" to listen to the generated voice.

Advanced Steps to Make your own voice cloning system:
1. Install Coqui AI TTS module using "pip install TTS"
2. Clone the Suno Bark code at huggingface using "git clone https://huggingface.co/suno/bark" command. Remember that you should be in the project folder. This step would create a bark folder and copy all the files from the hugging face reporsitory.
3. Now create a folder "bark_voices" at the root level in which again create a folder call "speaker". The "speaker" folder will be used to store your audio samples.
4. 

Note: You can also set "SUNO_USE_SMALL_MODELS=True" if you are using the CPU to make things a little faster OR you can use a smaller model call "bark-small".

