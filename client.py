import requests


BASE_URL = "http://localhost:8000"


def main():
    requests.post(f'{BASE_URL}/reset/')
    
    URLS = [
        "https://raw.githubusercontent.com/ollama/ollama/main/docs/faq.md",
    ]
    for url in URLS:
        requests.post(f'{BASE_URL}/scrape/', json={
            "url": url,
        })
    
    while True:        
        try:
            question = input('> Question: ')
        except KeyboardInterrupt:
            break
        
        response = requests.post(f'{BASE_URL}/ask/', json={
            "question": question,
        })
        response_data = response.json()
        is_ok = response_data.get('ok')
        
        if is_ok:
            print('> Answer:', response_data['answer'])
        else:
            print('> Error:', response_data)
        print()


if __name__ == "__main__":
    main()
