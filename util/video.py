import cv2 as cv

class Video():

    video = None
    name = None

    def __init__(self, _name):
        self.name = _name

    def frames(self, video_format='mp4'):
        '''
        a generator which reads a video file and yield each frame until its end.
        '''

        self.video = cv.VideoCapture('.'.join([self.name, video_format]))

        while True:
            if(self.video.get(cv.CAP_PROP_POS_FRAMES) == self.video.get(cv.CAP_PROP_FRAME_COUNT)):
                break
            ret, frame = self.video.read()
            del ret
            yield frame

    def release_video(self):
        print("<video> video released.")
        self.video.release()