data_params = {
    "start": 0.5,
}


def simple_ai_description(input_value, hold):
    feed_back_system = 0.4
    if hold != feed_back_system:
        data_params["final"] = feed_back_system
        if input_value > feed_back_system:
            return "Високе значення: Висока тригерна точка"
        else:
            return "Низьке значення: Низька тригерна точка"
    else:
        data_params["final"] = hold
        if input_value > hold:
            return "Високе значення: Висока тригерна точка"
        else:
            return "Низьке значення: Низька тригерна точка"


values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for value in values:
    resp = simple_ai_description(value, data_params["start"])

if data_params["start"] != data_params["final"]:
    print("Значення змінилося")
    print('Потрібно виставити новен стартове значення')


