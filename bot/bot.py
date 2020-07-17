import logging

from aiogram import Bot, Dispatcher, executor, types

from chatgame.games.guess_topic_game import GuessTopicGame, example_of_using_GuessTopicGame

API_TOKEN = "1320827829:AAE3eWAgjBQ5HGrG80uUCTO-65-FtNXmKVY"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

GAMES = {"1": GuessTopicGame}


@dp.message_handler(commands=['start'])
async def greetings(message: types.Message):
    """
    This handler will be called when user sends `/start` or command
    """
    # TODO Сделать набор кнопки для каждой из игр
    await message.reply("Hi!\nWhat game shall we play?.")


@dp.message_handler()
async def start_game(message: types.Message):
    """
    В зависимости от нажатой кнопки в функции greetings здесь запускается
    одна из игр, определенных как машины состояний в словаре GAMES
    :param message:
    :return:
    """
    game_class = GAMES[message.text]
    game = game_class()
    example_of_using_GuessTopicGame(game)
    await bot.send_message(message.chat.id, game.pert_gen_texts[0])

    # old style:
    # await bot.send_message(message.chat.id, message.text)

    # await message.answer(message.text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
