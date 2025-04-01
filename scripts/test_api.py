import requests

def test_predict_case(data, expected_key="survival_prediction"):
    url = "http://127.0.0.1:8000/predict/"
    response = requests.post(url, json=data)
    result = response.json()
    print("Input:", data)
    print("Response:", result)
    assert expected_key in result, f"Expected key '{expected_key}' not found in response."

if __name__ == "__main__":
    # Тестовый случай 1
    data1 = {"Pclass": 3, "Age": 25, "Fare": 7.25}
    test_predict_case(data1)

    # Тестовый случай 2
    data2 = {"Pclass": 1, "Age": 40, "Fare": 100.0}
    test_predict_case(data2)

    # Тестовый случай 3
    data3 = {"Pclass": 2, "Age": 30, "Fare": 50.0}
    test_predict_case(data3)

    print("Все тесты успешно пройдены!")
