import requests


def test_prompt_endpoint():
    url = 'http://localhost:4002'
    # Test the prompt endpoint
    data = [
        {
            "input": "How would a typical person answer each of the following questions about causation?\nA machine is set up in such a way that it will short circuit if both the black wire and the red wire touch the battery at the same time. The machine will not short circuit if just one of these wires touches the battery. The black wire is designated as the one that is supposed to touch the battery, while the red wire is supposed to remain in some other part of the machine. One day, the black wire and the red wire both end up touching the battery at the same time. There is a short circuit. Did the black wire cause the short circuit?\nOptions:\n- Yes\n- No", 
            "target": "No"
        }
    ]
    response = requests.post(url + '/prompt', json={'data': data})
    response_json = response.json()
    print(response_json)

    # Use pytest's assertion style
    assert 'prompt' in response_json, "Response does not contain 'prompt'"
    assert 'score' in response_json, "Response does not contain 'score'"
    assert isinstance(response_json['prompt'], str), "'prompt' is not a string"
    assert isinstance(response_json['score'], (int, float)), "'score' is not a number"
    assert len(response_json['prompt']) > 0, "'prompt' is empty"

def main():
    #run test:
    test_prompt_endpoint()
    
if __name__=='__main__':
    main()