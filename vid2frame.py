import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

import csv
import glob
import os
import os.path
from subprocess import call

def extract_files():

    data_file = []
    folders = ['Downloads\\Compressed\\datas\\']
   
    for vid_class in folders:
        class_files = glob.glob(vid_class + '/*.mp4')

        for video_path in class_files:
            # Get the parts of the file.
            video_parts = get_video_parts(video_path)

            filename_no_ext, filename = video_parts
           
            createFolder('.\\'+filename_no_ext+'\\')
            #print(os.path.exists('.\\'+filename_no_ext+'\\'))
            if not check_already_extracted(video_parts):
                # Now extract it.
                src = r'Downloads\\Compressed\\datas\\'+ filename # change the location to wherever you put your video
                
                dest = r'Downloads\\Compressed\\datas\\'+filename_no_ext+'\\'+filename_no_ext + '-%04d.jpg'
                call(["ffmpeg", "-i", src, dest]) # change the location to wherever you want put your video's frames

            # Now get how many frames it is.
            nb_frames = get_nb_frames_for_video(video_parts)

            data_file.append([filename_no_ext, nb_frames])

            print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    
    filename_no_ext, _ = video_parts
    generated_files = glob.glob(r'Downloads\\Compressed\\datas\\'+filename_no_ext+'\\' +\
                                filename_no_ext + '*.jpg')
    return len(generated_files)
def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('\\')
    filename = parts[-1]
    filename_no_ext = filename.split('.')[0]
    #classname = parts[2]
    #train_or_test = parts[1]
    #print(filename)
    return filename_no_ext, filename
def check_already_extracted(video_parts):
  
    filename_no_ext, _ = video_parts
    return bool(os.path.exists(r'Downloads\\Compressed\\datas\\'+ filename_no_ext + '-0001.jpg'))
def main():
    extract_files()
if __name__ == '__main__':
    main()



















