import requests


def main():
    URLS = [
        "https://raw.githubusercontent.com/ollama/ollama/main/docs/faq.md",
    ]
    for url in URLS:
        requests.post('http://localhost:8001/scrape/', json={
            "url": url,
        })
    
    while True:        
        question = input('> Question: ')
        
        response = requests.post('http://localhost:8001/ask/', json={
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
