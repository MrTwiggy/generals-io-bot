# AlphaGenerals - A Generals.io AI Agent
Developed by **Ty Sayers & Sonny Li**

This is a machine learning based agent that plays the game [generals.io.](http://bot.generals.io) 

Currently employs behavioural cloning along with data augmentation that build on convolutional neural networks as their base models to mimic player actions.

The framework for environment interaction through server sockets is forked from the TNT generals bot. The underlying structures used for TNT is Toshima's base generals web client.

## Replays Of The Bot In Action
[21/04/2017 Version - Behavioural Cloning Only](http://bot.generals.io/replays/Sx01pGdAe)

## Install Requirements
To install all requirements, run the command `pip install -r requirements.txt`

## Train AlphaGenerals Agent
- Run command `python ./train_imitation.py [ReplayDirectory] [ModelName] [MaxReplaysToLoad]`

## Configure & Run Agent
- Set the `USER_ID, USER_NAME` in **config.py** for your bot account.
- Run command 'sh runBot.sh [BotType] [ModelName]'

## Credits
[Toshima's Generals.io Client](https://github.com/toshima/generalsio) - Provides the underlying socket-level server interactions and move handling with the live game servers.

[Tim-Hub's TNT Bot](https://github.com/tim-hub/generals.io-python-bot-TNT) - Provided the basic structure for bot interaction with Toshima's client.

[Generals.io Replay Utils](https://github.com/vzhou842/generals.io-Replay-Utils) - Provides ability to convert serialized .gior files into a proper plain-text JSON .gioreplay format.