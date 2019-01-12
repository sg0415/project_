import pygame
from pygame.locals import *
import sys
import pyaudio
from keras.models import load_model
import wave
import numpy as np
from array import array
from preprocess import *
import time
import threading
import random

# input 모양
feature_dim_1 = 20
channel = 1
feature_dim_2 = 11


# 색깔
black = (0, 0, 0)
white = (255, 255, 255)

red = (200, 0, 0)
green = (0, 200, 0)

bright_red = (255, 0, 0)
bright_green = (0, 255, 0)


pygame.init()

#화면 크기
width = 640
height = 480

#화면 설정
screen = pygame.display.set_mode((width, height), DOUBLEBUF)

# 창에 제목
pygame.display.set_caption('pet_game')

clock = pygame.time.Clock()


#버튼 생성시에 사용되는 함수
def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect()

#버튼생성 함수
def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()  #마우스 위치 받아들이는 함수
    click = pygame.mouse.get_pressed()  #마우스 클릭받는 함수
    string = 'none'
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(screen, ac, (x, y, w, h))  #버튼 범위에 들어왔을때
        if click[0] == 1 and action != None:
            if action == 'game_loop':   #게임 인트로 사용시 필요한 함수 1
                gameloop()
            elif action == 'quitgame':
                pygame.quit()
                quit()
            elif action == 'speak': #우리가 사용하는 음성인식과정
                string = speak()
                return string


    else:
        pygame.draw.rect(screen, ic, (x, y, w, h))
    #버튼의 모양과 내용 만드는 부분
    smallText = pygame.font.SysFont("comicsansms", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    screen.blit(textSurf, textRect)
    return string


def happy_name(msg, x, y, w, h, ic):
    pygame.draw.rect(screen, ic, (x, y, w, h))
    smallText = pygame.font.SysFont("comicsansms", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    screen.blit(textSurf, textRect)

def game_intro():
    intro = True
    while intro:
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.quit()

        screen.fill(bright_green)
        largeText = pygame.font.SysFont("comicsansms", 100)
        TextSurf, TextRect = text_objects("pet game", largeText)
        TextRect.center = ((width / 2), (height / 2))
        screen.blit(TextSurf, TextRect)
        dog_group.update('stand')
        dog_group.draw(screen)

        button("GO!", 100, 400, 100, 50, green, bright_green, 'game_loop')
        button("Quit", 500, 400, 100, 50, red, bright_red, 'quitgame')

        pygame.display.update()
        clock.tick(15)


def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape( 1,feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]

def action(string) :
    largeText = pygame.font.SysFont("comicsansms", 100)
    TextSurf, TextRect = text_objects(string, largeText)
    TextRect.center = ((width / 2), (height / 2))
    screen.blit(TextSurf, TextRect)



def speak() :


    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "file.wav"

    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        vol = max(data_chunk)
        frames.append(data)

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("end")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



    model = load_model('epoch50_class9_model_bgnoise.h5')


#    action(string)
    return predict(WAVE_OUTPUT_FILENAME, model)


#이미지를 불러오는 클래스
class Spritesheet(object):
    """ Class used to grab images out of a sprite sheet. """

    def __init__(self, file_name):
        """ Constructor. Pass in the file name of the sprite sheet. """

        # Load the sprite sheet.
        self.sprite_sheet = pygame.image.load(file_name).convert()

    def get_image(self, x, y, width, height):
        """ Grab a single image out of a larger spritesheet
            Pass in the x, y location of the sprite
            and the width and height of the sprite. """

        # Create a new blank image
        image = pygame.Surface([width, height]).convert()

        # Copy the sprite from the large sheet onto the smaller image
        image.blit(self.sprite_sheet, (0, 0), (x, y, width, height))

        # Assuming black works as the transparent color
        image.set_colorkey(white)

        # Return the image
        return image

class hp_status():
    def __init__(self, hp, list):
        self.hp_img = hp
        self.hp_list = list
        for i in range(1, 4):
            self.hp_list.append((self.hp_img).get_rect(x=(50*i), y=50))

    # 체력게이지
    def plus_hp(self):
        print("플러스 : ", len(self.hp_list))
        self.hp_list.append(self.hp_img.get_rect(x=(50+(len(self.hp_list))*50), y=50))

    def minus_hp(self):
        if(len(self.hp_list)>=1):
            del self.hp_list[len(self.hp_list)-1]
        else:
            print("갱얼쥐의 체력이 없어요...", len(self.hp_list))
    def update_hp(self):
        for i in range(0, len(self.hp_list)):
            screen.blit(self.hp_img, self.hp_list[i])
    def get_hp(self):
        return len(self.hp_list)


class jelly():
    def __init__(self, image):
        self.dog_jelly = image
        self.jelly_x = 100
        self.jelly_y = 350
        self.rect = self.dog_jelly.get_rect(x=self.jelly_x, y=self.jelly_y)

    def update_jelly(self):
        screen.blit(self.dog_jelly, (self.jelly_x, self.jelly_y))

    def collision_check(self, dog_x, y, dog_hp):
        #print("텟트")
        if self.jelly_x < dog_x:
        #    print("jelly_x : ", self.jelly_x, "강아지 x : ", dog_x)
            if self.jelly_x+10 == dog_x:
                print("충돌")
                dog_hp.plus_hp()
                self.jelly_x = -100
                threading.Timer(3, self.set_jelly).start()
        elif self.jelly_x > dog_x:
        #    print("jelly_x : ", self.jelly_x, "강아지 x : ", dog_x)
            if self.jelly_x == dog_x+200:
                print("충돌")
                dog_hp.plus_hp()
                self.jelly_x = -100
                threading.Timer(10, self.set_jelly()).start()

    def set_jelly(self):
        #
        #timer = threading.Timer(2, self.set_jelly)
        print("test...")
        self.jelly_x = 5*random.randrange(1, 115)
        #timer.start()
        #self.update_jelly()

class MySprite_4(pygame.sprite.Sprite):
    def __init__(self, file1,file2,file3,file4,file5,file6,file7,file8,file9,file10):
        super(MySprite_4, self).__init__()
        # adding all the images to sprite array
        #강아지 위치변수들
        self.x = 200
        self.y = 200
        #강아지 크기변수들
        self.size_x = 200
        self.size_y= 200


        #이미지 불러와서 여러장인 경우 나누어 array구성
        sprite = Spritesheet(file1)
        self.images_stand = []
        image = sprite.get_image(0,0,200,200)
        image = pygame.transform.scale(image,(self.size_x,self.size_y))
        self.images_stand.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_stand.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_stand.append(image)
        image = sprite.get_image(600, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_stand.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_stand = 0

        sprite = Spritesheet(file2)
        self.images_wow = []
        image = sprite.get_image(0,0,200,200)
        image = pygame.transform.scale(image,(self.size_x,self.size_y))
        self.images_wow.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_wow.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_wow.append(image)
        image = sprite.get_image(600, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_wow.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_wow = 0

        sprite = Spritesheet(file3)
        self.images_hi = []
        image = sprite.get_image(0,0,200,200)
        image = pygame.transform.scale(image,(self.size_x,self.size_y))
        self.images_hi.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_hi.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_hi.append(image)
        image = sprite.get_image(600, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_hi.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_hi = 0

        sprite = Spritesheet(file4)
        self.images_left = []
        image = sprite.get_image(0,0,200,200)
        image = pygame.transform.scale(image,(self.size_x,self.size_y))
        self.images_left.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_left.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_left.append(image)

        # index value to get the image from the array
        # initially it is 0
        self.index_left = 0

        sprite = Spritesheet(file5)
        self.images_right = []
        image = sprite.get_image(0,0,200,200)
        image = pygame.transform.scale(image,(self.size_x,self.size_y))
        self.images_right.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_right.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x,self.size_y))
        self.images_right.append(image)
                # index value to get the image from the array
        # initially it is 0
        self.index_right = 0

        sprite = Spritesheet(file6)
        image = sprite.get_image(0,0,200,200)
        self.images_stand_left = pygame.transform.scale(image,(self.size_x,self.size_y))


        sprite = Spritesheet(file7)
        image = sprite.get_image(0,0,200,200)
        self.images_stand_right = pygame.transform.scale(image,(self.size_x,self.size_y))

        sprite = Spritesheet(file8)
        self.images_up = []
        image = sprite.get_image(0, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_up.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_up.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_up = 0

        sprite = Spritesheet(file9)
        self.images_down = []
        image = sprite.get_image(0, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_down.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_down.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_down = 0


        sprite = Spritesheet(file10)
        self.images_bed = []
        image = sprite.get_image(0, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_bed.append(image)
        image = sprite.get_image(200, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_bed.append(image)
        image = sprite.get_image(400, 0, 200, 200)
        image = pygame.transform.scale(image, (self.size_x, self.size_y))
        self.images_bed.append(image)
        # index value to get the image from the array
        # initially it is 0
        self.index_bed = 0



        # now the image that we will display will be the index from the image array
        self.image = self.images_stand[self.index_stand]

        # creating a rect at position x,y (5,5) of size (150,198) which is the size of sprite
        self.rect = pygame.Rect(self.x, self.y, 15, 20)

    #강아지 모양 업데이트
    def update(self,string='stand',status = 'go'):
        if status == 'go':

            if string == 'go':  #stand를 사용
                # when the update method is called, we will increment the index
                self.index_stand += 1
                # if the index is larger than the total images
                if self.index_stand >= len(self.images_stand):
                    # we will make the index to 0 again
                    self.index_stand = 0

                # finally we will update the image that will be displayed
                self.image = self.images_stand[self.index_stand]

            if string == 'wow':
                # when the update method is called, we will increment the index
                self.index_wow += 1
                # if the index is larger than the total images
                if self.index_wow >= len(self.images_wow):
                    # we will make the index to 0 again
                    self.index_wow = 0

                # finally we will update the image that will be displayed
                self.image = self.images_wow[self.index_wow]

            if string == 'happy':
                # when the update method is called, we will increment the index
                self.index_hi += 1
                # if the index is larger than the total images
                if self.index_hi >= len(self.images_hi):
                    # we will make the index to 0 again
                    self.index_hi = 0


                # finally we will update the image that will be displayed
                self.image = self.images_hi[self.index_hi]

            if string == 'left':
                # when the update method is called, we will increment the index
                self.index_left += 1
                if self.x >= 0:
                    self.x -= 5
                # if the index is larger than the total images
                if self.index_left >= len(self.images_left):
                    # we will make the index to 0 again
                    self.index_left = 0
                # finally we will update the image that will be displayed
                self.image = self.images_left[self.index_left]

            if string == 'right':
                # when the update method is called, we will increment the index
                self.index_right += 1
                if self.x < (width - self.size_x):
                    self.x += 5
                # if the index is larger than the total images
                if self.index_right >= len(self.images_right):
                    # we will make the index to 0 again
                    self.index_right = 0
                # finally we will update the image that will be displayed
                self.image = self.images_right[self.index_right]

            if string == 'up':
                # when the update method is called, we will increment the index
                self.index_up += 1
                # if the index is larger than the total images
                if self.index_up >= len(self.images_up):
                    # we will make the index to 0 again
                    self.index_up = 0
                self.image = self.images_up[self.index_up]
            if string == 'down':
                # when the update method is called, we will increment the index
                self.index_down += 1
                # if the index is larger than the total images
                if self.index_down >= len(self.images_down):
                    # we will make the index to 0 again
                    self.index_down = 0
                self.image = self.images_down[self.index_down]
            if string == 'bed':
                # when the update method is called, we will increment the index
                self.index_bed += 1
                # if the index is larger than the total images
                if self.index_bed >= len(self.images_bed):
                    # we will make the index to 0 again
                    self.index_bed = 0
                self.image = self.images_bed[self.index_bed]

            if string == 'left_up':
                self.image = self.images_stand_left

            if string == 'right_up':
                self.image = self.images_stand_right


            self.rect = pygame.Rect(self.x, self.y, 15, 20)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y



def gameloop ():

    '''
    # 이미지 불러와서 크기 조정

    dog = pygame.image.load('dog1.png')
    dog = pygame.transform.scale(dog, (100, 100))

    # 초기 위치
    dogx = 200
    dogy = 200

    screen.blit(dog, (48, 32))'''


    # 배경화면
    background = pygame.image.load('dog/back_image.jpg')
    background= pygame.transform.scale(background,(640,480))
    screen.blit(background, (0, 0))
    #우리가 사용할 강아지 인스턴스 생성
    dog=MySprite_4('dog/stand.png','dog/wow.png','dog/hi.png','dog/move_left.png','dog/move_right.png','dog/up_left.png','dog/up_right.png','dog/up.png','dog/down.png','dog/bed.png')
    dog_group = pygame.sprite.Group(dog)
    string = 'stand'
    status = 'go'
    hp_image = pygame.image.load('dog/heart.png')
    ls1 = []
    dog_hp = hp_status(hp_image, ls1)
    jelly_image = pygame.image.load('dog/jelly.png')
    item_jelly = jelly(jelly_image)

    while True:


        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            #테스트위한 입력
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_UP:
                    string = 'up'
                if event.key == pygame.K_LEFT:
                    string='left'
                if event.key == pygame.K_RIGHT:
                    string='right'
                if event.key == pygame.K_DOWN:
                    string= 'down'
                if event.key == pygame.K_1:
                    string ='happy'
                if event.key == pygame.K_2:
                    string = 'wow'
                if event.key == pygame.K_3:
                    string = 'bed'
                if event.key == pygame.K_4:
                    string ='go'

                if event.key == pygame.K_SPACE:
                    if dog_hp.get_hp() == 0:
                        print("강아지의 체력이 없어요...")
                    else:
                        if string=='left':
                            string = 'left_up'
                            dog_group.update(string,status)
                            dog_group.clear(screen, background)
                            dog_group.draw(screen)
                            dog_hp.minus_hp()
                            dog_hp.update_hp()
                            pygame.display.update()

                        elif string == 'right':
                            string='right_up'
                            dog_group.update(string,status)
                            dog_group.clear(screen, background)
                            dog_group.draw(screen)
                            dog_hp.minus_hp()
                            dog_hp.update_hp()
                            pygame.display.update()

                        else :
                            print("stand...")
                            dog_group.update(string,status)
                            dog_group.clear(screen, background)
                            dog_group.draw(screen)
                            dog_hp.minus_hp()
                            dog_hp.update_hp()
                            pygame.display.update()
                        string=speak()
                    if string == 'stop':
                        status = 'stop'
                    elif string == 'go':
                        status = 'go'
                    print(string,status)

        dog_group.update(string,status)
        screen.blit(background, (0, 0))
        dog_group.draw(screen)
        item_jelly.update_jelly()
        item_jelly.collision_check(dog.get_x(),dog.get_y(),dog_hp)
        button("Press Space", 500, 50, 125, 50, green, bright_green,'speak')
        happy_name("happy", dog.get_x()+75, dog.get_y()+200, 75, 30, white)
        dog_hp.update_hp()
        pygame.display.update()
        clock.tick(5)


#game_intro()
gameloop()
