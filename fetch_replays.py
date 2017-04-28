# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:37:43 2017

@author: MrTwiggy
"""
#import lxml.html
#from lxml.cssselect import CSSSelector
from lxml import html
import requests
import json
import sys
import multiprocessing
import numpy as np
import os
import urllib.request
import shutil

# Example usage: python fetch_replays.py ./replays/MrTwiggy 4 MrTwiggy 5000

REPLAY_REQUEST = 'http://generals.io/{}.gior'
REPLAYS_BY_USERNAME = 'http://generals.io/api/replaysForUsername?u={}&offset=0&count={}'

arg_count = len(sys.argv) - 1
REPLAY_FOLDER = sys.argv[1] if arg_count >= 1 else "./replays"
THREAD_COUNT = int(sys.argv[2]) if arg_count >= 2 else 8
USER_NAME = sys.argv[3] if arg_count >= 3 else "[Bot] FloBot"
REPLAY_LIMIT = int(sys.argv[4]) if arg_count >= 4 else 50

#THREAD_COUNT = 5

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def download_replays(threadID, replayFolder, replayNames):
    #print("test")
    for i in range(len(replayNames)):
        print("Thread {}: Requesting and downloading replay {} out of {}".format(threadID, i, len(replayNames)))
        replayName = replayNames[i]
        fileName = "{}/{}.gior".format(replayFolder, replayName)
        
        if os.path.isfile(fileName):
            continue
        
        replay_url = REPLAY_REQUEST.format(replayName)
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        
        headers={'User-Agent':user_agent,}
        request=urllib.request.Request(replay_url,None,headers) #The assembled request
        with urllib.request.urlopen(request) as response, open(fileName, 'wb') as out_file:
            data = response.read() # a `bytes` object
            out_file.write(data)

def download_player_replays(userId, replayFolder = "./replays", gameLimit = 50):
    gameIds = requests.get(REPLAYS_BY_USERNAME.format(userId, gameLimit)).json()
    #page = requests.get(PROFILE_URL.format(userId))
    #tree = html.fromstring(page.content)
    replayNames = []
    for i in range(len(gameIds)):
        if gameIds[i]['type'] == '1v1':
            replayNames.append(gameIds[i]['id'])
    #replayNames = [gameIds[i]['replayName'] for i in range(len(gameIds))]
    #print(replayNames)
    #print(replayNames)
    print("1v1 replays located: {}".format(len(replayNames)))
    replayIndices = np.array_split(np.arange(len(replayNames)), THREAD_COUNT)
    replaySets = [[] for i in range(THREAD_COUNT)]
    for i in range(THREAD_COUNT):
        for index in replayIndices[i]:
            replaySets[i].append(replayNames[index])
    threads = []
    #print(replaySets)
    for threadID in range(THREAD_COUNT):
        thread = multiprocessing.Process(target=download_replays, args = (threadID, replayFolder, replaySets[threadID]))   
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    for threadId in range(THREAD_COUNT):
        print("Joining on thread {}".format(threadId))
        threads[threadId].join()

#download_player_replays(2697, "./test_replays", 10)
download_player_replays(USER_NAME, REPLAY_FOLDER, REPLAY_LIMIT)