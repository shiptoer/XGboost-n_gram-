from pydantic import ValidationError

import api
from project_utils import sanitize_csv_cell


def main():
    model, vectorizer = api.load_model()
    assert model is not None, "Model file could not be loaded"
    assert vectorizer is not None, "Vectorizer file could not be loaded"

    malicious_vec = vectorizer.transform(["1 OR 1=1"])
    malicious_prediction = int(model.predict(malicious_vec)[0])
    assert malicious_prediction == 1, "Expected SQL injection payload to be detected"

    benign_vec = vectorizer.transform(["SELECT name FROM products WHERE id = 5"])
    benign_prediction = int(model.predict(benign_vec)[0])
    assert benign_prediction == 0, "Expected benign SQL query to be allowed"

    try:
        api.PayloadRequest(payload="x" * (api.MAX_PAYLOAD_LENGTH + 1))
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected payload max_length validation to fail")

    assert sanitize_csv_cell("=1+1") == "'=1+1"
    assert sanitize_csv_cell("normal payload") == "normal payload"

    print("smoke tests ok")


if __name__ == "__main__":
    main()
