# AlphaGenerals - A Generals.io AI Agent
Developed by **Ty Sayers & Sonny Li**

This is a machine learning based agent that plays the game [generals.io.](http://bot.generals.io) 

Currently employs behavioural cloning along with data augmentation that build on convolutional neural networks as their base models to mimic player actions.

The framework for environment interaction through server sockets is forked from the TNT generals bot. The underlying structures used for TNT is Toshima's base generals web client.

## Replays Of The Bot In Action
[21/04/2017 Version - Behavioural Cloning Only] (http://bot.generals.io/replays/Sx01pGdAe)

## Install Requirements
`pip install -r requirements.txt`

## Train AlphaGenerals Agent
- Train model with command `python ./train_imitation.py [ReplayDirection] [ThreadCount] [AgentName] [MaxReplaysToLoad]`

## Configure & Run Agent
- Set the `USER_ID, USER_NAME, GAME_ID (for custom game)` in **config.py**.
- Choose the bot to run, for example `python bot_TNT.py`

## Credits
[Toshima's Generals.io Client](https://github.com/toshima/generalsio)

[Tim-Hub's TNT Bot] (https://github.com/tim-hub/generals.io-python-bot-TNT)

