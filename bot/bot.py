import logging
from aiogram import Bot, Dispatcher, executor, types
from chatgame.games.guess_topic_game import GuessTopicGame, example_of_using_GuessTopicGame
from bot.post_processing import *
# guess_game_first_post_processing, guess_game_second_post_processing,CALLBACK_IN_GUESS_TOPIC_GAME

API_TOKEN = "1320827829:AAE3eWAgjBQ5HGrG80uUCTO-65-FtNXmKVY"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

GAMES = {"1": GuessTopicGame}
GAMES_NAMES = {'1': 'Guess topic'}

CALLBACK_SELECT_GAME = 'select_game_'
CALLBACK_USER_ANSWER = {'1': CALLBACK_IN_GUESS_TOPIC_GAME}

GAMES_FUNCTION_FIRST_POST_PROCESSING = {'1': guess_game_first_post_processing}
GAMES_FUNCTION_SECOND_POST_PROCESSING = {'1': guess_game_second_post_processing}


@dp.message_handler(commands=['start'])
async def greetings(message: types.Message):
    """
    This handler will be called when user sends `/start` or command
    """
    # TODO Добавлять сюда кнопки для новых игр

    choose_games_keyboard = types.InlineKeyboardMarkup(resize_keyboard=True,
                                                       one_time_keyboard=True,
                                                       reply=False)
    button_guess_topic_game = types.InlineKeyboardButton('Guess topic',
                                                         callback_data=CALLBACK_SELECT_GAME + '1')
    choose_games_keyboard.add(button_guess_topic_game)

    await message.reply("Hi!\nWhat game shall we play?",
                        reply_markup=choose_games_keyboard,
                        reply=False)


@dp.callback_query_handler(lambda c: c.data and c.data.startswith(CALLBACK_SELECT_GAME))
async def start_game(callback_query: types.CallbackQuery):
    """
    В зависимости от нажатой кнопки в функции greetings здесь запускается
    одна из игр, определенных как машины состояний в словаре GAMES
    :param callback_query:
    :return:
    """
    code_of_game = callback_query.data[len(CALLBACK_SELECT_GAME):]

    # Скрываем кнопки после выбора игры
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)

    await callback_query.message.reply("You choose the '{}' game!".format(GAMES_NAMES[code_of_game]),
                                       reply=False)

    game_class = GAMES[code_of_game]

    # TODO Здесь надо принять от пользователя затравку для текста, если игра это предполагает

    game = game_class()
    game.triger_start_game()  # TODO Затравка передается так: (conditional_text_prefix='The world ')

    # Отправляем сгенерированный текст пользователю
    game.triger_receive_text()

    await callback_query.message.reply(game.pert_gen_texts[0],
                                       reply=False)

    # В зависимости от игры вызываем свою функцию
    post_processing_first = GAMES_FUNCTION_FIRST_POST_PROCESSING[code_of_game]

    await post_processing_first(game, callback_query)

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith(CALLBACK_USER_ANSWER[code_of_game]))
    async def check_answer(callback_query_inside: types.CallbackQuery):

        post_processing_second = GAMES_FUNCTION_SECOND_POST_PROCESSING[code_of_game]
        answer = await post_processing_second(bot, CALLBACK_USER_ANSWER[code_of_game], callback_query_inside)

        game.triger_finish_game(answer)

        await callback_query.message.reply("You {0} the game.".format(game.state[5:]),
                                           reply=False)


if __name__ == '__main__':
    executor.start_polling(dp,
                           skip_updates=True)
