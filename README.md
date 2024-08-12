# Video Translator Documentation
## About
MrBeast and other YouTubers make a ton of money translating their videos into other languages for a wider audience of people to enjoy, but this can be expensive because you have to hire voice actors that sound like the people in the video. With the power of AI, I created a system that allows  people to dub their videos for free so they can go out to a wider audience. The script can mimic the sound and energy of the original voice in a different language.
## Requirements
* This script was tested with [Python 3.10.11](https://www.python.org/downloads/windows/#:~:text=Python%203.10.11). It might also work with other Python versions, I haven't bothered to check. Make sure you've [added this version to your PATH variable](https://www.youtube.com/watch?v=iNoQeRj52zo) if it didn't happen automatically. To check if Python is in your PATH variable, open Windows PowerShell and type `python -V`. It should return `Python 3.10.11`.
* Install [FFMPEG](https://www.ffmpeg.org/download.html) for video processing.
* I use the [GitHub CLI](https://cli.github.com) to clone the repository, which I reccomment. You can also accomplish the same thing by going to code > download zip on the GitHub website and extract it into the folder you create in step 2.
## How to use FOR BEGINNERS: Windows
1. Create a folder to use the project in. I'll create a folder in my Documents folder called `translator`
2. Open Windows PowerShell and type `cd C:\Users\(user)\Documents\translator` to open the project folder. Replace the path with the path of your created folder.
3. Type `gh repo clone KaiSereni/Video-Translator` to clone the repository into your folder. If it asks you to login, keep pressing enter to use all the default options and then log in or sign up in your browser. If it's successful, retype the command.
4. Then, run `cd Video-Translator` to open the folder.
5. Type `python -m venv venv` to create a virtual environment. Once this finishes, type `venv/Scripts/activate` to activate the environment. There will now be a venv indicator in your shell.
6. Type `pip install -r requirements.txt` to install the required packages. This will take around 3 minutes.
7. In your file explorer, copy the video you want to translate into the `src` folder, which should be at `C:\Users\(user)\Documents\translator\Video-Translator\src` (I added a demo video, which can be deleted). Rename this file to `input.mp4`. IMPORTANT: Make sure the video has no music or sound effects, only speech. You can add the music and sound in post. The video must also be in mp4 format.
8. In your file explorer, open the `output_language.txt` file, which should be at  `C:\Users\(user)\Documents\translator\Video-Translator\output_language.txt`. This is where you specify the language you want to translate **into**. Type the two-letter language code and nothing else. For example: type `en` for english, `es` for spanish, `fr` for french, or `de` for german. [Here](https://www.w3schools.com/tags/ref_language_codes.asp#:~:text=ISO%20639)'s a full list of language codes. Save the text file and close your text editor.
9. Finally, to translate the video, go back to the PowerShell window and type `python main.py`. This could take between 1 and 30 minutes depending on the length of the video and whether you've use the script before. The final product will be in the `Video-Translator` folder.
## Quickstart for programmers
1. Clone the repo and open it in a shell.
2. Create a virtual environment in that folder with `python -m venv venv`
3. Install the requirements with `pip install -r requirements.txt`
4. Open the `output_language.txt` file and type the two-letter output language code, and nothing else. For example: type `en` for english, `es` for spanish, `fr` for french, or `de` for german. [Here](https://www.w3schools.com/tags/ref_language_codes.asp#:~:text=ISO%20639)'s a full list of language codes.
5. copy the video you want to translate into the `Video-Translator/src` folder. Rename this file to `input.mp4`. IMPORTANT: Make sure the video has no music or sound effects, only speech. You can add the music and sound in post. The video must also be in mp4 format.
6. Run `python main.py` to translate the video. This can take between 1 and 30 minutes depending on the video length and whether you've used the script before (if models have already downloaded).
<br>**Note:** This repo should work out-of-the-box, as there's a demo video in the src directory. You can just run `main.py` to test the repo. Also, if you have Cuda installed, this works faster with Cuda.

## Future Goals
* Audio chunk time remapping
* Better text-to-speech model

**This repo was created by Pohakoo, LLC. See more projects at [forgotai.com](www.forgotai.com).**