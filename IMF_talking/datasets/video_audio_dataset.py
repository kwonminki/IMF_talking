import os
import torch
from easy_video import EasyReader, EasyWriter
from easy_video.utils import mp4list
import random

def video_audio_collate_fn(batch):
    pass

class VideoAudioDataset(torch.utils.data.Dataset):
    def __init__(self, video_folder_list_txt_file, recompute=False, 
                 all_video_suffix='_all_video_files.txt',
                 clean_video_suffix='_clean_video_files.txt',
                 n_frames=-1, # number of frames to sample from each video. if -1, sample all frames,
                 target_fps=None, # if not None, resample the video to this fps
                 target_resolution=None, # if not None, resize the video to this resolution
                 audio_sample_rate=16000, # sample rate of the audio
                 audio_nchannels=1, # number of audio channels
                 ):

        assert video_folder_list_txt_file.endswith('.txt')
        assert all_video_suffix.endswith('.txt')

        self.video_folder_list_txt_file = video_folder_list_txt_file
        self.all_video_files_file = self.video_folder_list_txt_file.replace('.txt', all_video_suffix)
        self.clean_video_files_file = self.video_folder_list_txt_file.replace('.txt', clean_video_suffix)

        if recompute: # recompute the list of video files
            self.all_video_files = self._compute_all_video_files()
        elif not os.path.exists(self.all_video_files_file): # if the file does not exist, compute it
            self.all_video_files = self._compute_all_video_files()

        elif os.path.exists(self.clean_video_files_file): # if the clean file exists, use it
            with open(self.clean_video_files_file, 'r') as f:
                self.all_video_files = f.readlines()
        elif not os.path.exists(self.clean_video_files_file): # if the clean file does not exist, use the all file
            with open(self.all_video_files_file, 'r') as f:
                self.all_video_files = f.readlines()

        self.n_frames = n_frames
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.audio_sample_rate = audio_sample_rate
        self.audio_nchannels = audio_nchannels


    def _compute_all_video_files(self):
        all_video_files = []
        with open(self.video_folder_list_txt_file, 'r') as f:
            video_folders = f.readlines()

        for video_folder in video_folders:
            video_folder = video_folder.strip()
            video_files = mp4list(video_folder)
            all_video_files.extend(video_files)

        with open(self.all_video_files_file, 'w') as f:
            f.write('\n'.join(all_video_files))

        return all_video_files
    

    def _save_clean_video_files(self):
        with open(self.clean_video_files_file, 'w') as f:
            f.write('\n'.join(self.all_video_files))

    def __len__(self):
        return len(self.all_video_files)

    def __getitem__(self, idx):

        return_dict = {}

        while True:
            try:
                video_file = self.all_video_files[idx]
            except Exception as e:
                    print(f"Error: {e}")
                    print("Maybe all video files are already read. Randomly selecting a video file.")
                    idx = random.randint(0, len(self.all_video_files) - 1)
                    video_file = self.all_video_files[idx]

            try:
                reader = EasyReader(
                                    video_file,
                                    load_video=True,
                                    load_audio=True,
                                    target_video_fps=self.target_fps,
                                    target_resolution=self.target_resolution,
                                    audio_fps=self.audio_sample_rate,
                                    audio_nchannels=self.audio_nchannels,
                                    )
                
                if self.n_frames == -1:
                    start_idx = 0
                    end_idx = -1
                else:
                    total_frame_num = reader.n_frames
                    start_idx = random.randint(0, total_frame_num - self.n_frames - 1) # -1 for safety
                    end_idx = start_idx + self.n_frames

                video_array, audio_array = reader.get_video_array_audio_array(start=start_idx, end=end_idx)
                
                video_fps = reader.video_fps
                audio_fps = reader.audio_fps

                return_dict['video'] = video_array
                return_dict['audio'] = audio_array
                return_dict['video_fps'] = video_fps
                return_dict['audio_fps'] = audio_fps
                return_dict['video_file'] = video_file

                return return_dict
            
            except Exception as e:
                video_file = self.all_video_files[idx]
                print(f"Error: {e}")
                print(f"Error reading video file: {video_file}")
                self.all_video_files.remove(video_file)
                self._save_clean_video_files()
                idx = random.randint(0, len(self.all_video_files) - 1)
                video_file = self.all_video_files[idx]
                print(f"Trying another video file: {video_file}")
            
                    


if __name__ == "__main__":
    print("Testing VideoAudioDataset")

    os.makedirs('test', exist_ok=True)

    test_video_folders = [
        "/home/compu/Wholeprocess/GivernyAI/test/video_test_sample/video/face_bbox",
        "/home/compu/Wholeprocess/GivernyAI/test/doctor/docter_1/video/face_bbox",
    ]

    with open('video_folders.txt', 'w') as f:
        f.write('\n'.join(test_video_folders))

    dataset = VideoAudioDataset(video_folder_list_txt_file='video_folders.txt', recompute=False, n_frames=20, target_fps=30, target_resolution=(512, 512))

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Video file: {data['video_file']}")
        print(f"Video shape: {data['video'].shape}")
        print(f"Audio shape: {data['audio'].shape}")
        print(f"Video FPS: {data['video_fps']}")
        print(f"Audio FPS: {data['audio_fps']}")
        print("=====================================")

        EasyWriter.writefile(f'test/video_{i}.mp4', 
                             video_array= data['video'], 
                             audio_array=data['audio'],
                             video_fps=data['video_fps'], audio_fps=data['audio_fps'],
                             video_size=(512, 512))

        if i == 10:
            break

    os.remove('video_folders.txt')
    os.remove('video_folders_all_video_files.txt')
    os.remove('video_folders_clean_video_files.txt')