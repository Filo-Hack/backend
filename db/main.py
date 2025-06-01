from dbManager import ChromaDBManager
from ChatEngine import ChatEngine
from datetime import datetime
from random import  random


def input_db():
    # Базовая инициализация времени
    base_time = datetime(2025, 5, 31, 16, 0, 0)

    # Источники данных
    objects = [
        "стиральная машина", "настольная лампа", "штора", "кофемашина", "чайник",
        "входная дверь", "дверь в ванную", "дверь в гостиную"
    ]

    device_events = [
        "была включена", "была выключена", "была открыта", "была закрыта",
        "начала работу", "завершила цикл", "отправила уведомление"
    ]

    extra_facts = [
        "Я люблю принимать душ под музыку",
        "Я всегда пью чай перед сном",
        "Я не выхожу из дома без чашки кофе",
        "Я предпочитаю читать в тишине",
        "Я встаю каждый день в 6 утра",
        "Я не люблю яркий свет по вечерам",
        "Я слежу за температурой в комнате",
        "Я закрываю шторы, когда начинаю работать",
        "Я проверяю входную дверь перед сном",
        "Я включаю лампу при чтении"
    ]

    extra_requests = [
        "Как улучшить концентрацию?", "Как лучше высыпаться?",
        "Что делать, если нет мотивации?", "Что почитать вечером перед сном?",
        "Как избавиться от тревоги?", "Как сделать утро бодрым?"
    ]

    text_sources = {
        "fact": extra_facts,
        "request": extra_requests
    }

    # Генерация 100 записей
    entries_extended = []
    for i in range(100):
        choice = random.choices(["fact", "request", "device"], weights=[0.4, 0.4, 0.2])[0]

        if choice == "device":
            obj = random.choice(objects)
            event = random.choice(device_events)
            text = f"В умном доме {obj} {event}"
        else:
            text = random.choice(text_sources[choice])

        timestamp = (base_time + datetime.timedelta(minutes=i * 3)).isoformat()
        entry = {
            "text": text,
            "type_message": choice,
            "id": f"msg_{1000 + i}",
            "metadata": {
                "timestamp": timestamp
            }
        }
        entries_extended.append(entry)
    return entries_extended

if __name__ == "__main__":
    events = input_db()


    db = ChromaDBManager()
    chat = ChatEngine(chroma_db=db)
    for e in events:
        chat.save_record(e)
    question = "Что произошло на конференции?"
    answer = chat.generate_response(question)
    print(answer)



