import requests
import csv
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

page = requests.get('https://quotes.toscrape.com')

soup = BeautifulSoup(page.text, 'html.parser')

quotes = []

quote_elements = soup.find_all('div', class_='quote')

for quote_element in quote_elements:
    # extract the text of the quote
    text = quote_element.find('span', class_='text').text
    # extract the author of the quote
    author = quote_element.find('small', class_='author').text

    # extract the tag <a> HTML elements related to the quote
    tag_elements = quote_element.select('.tags .tag')

    # store the list of tag strings in a list
    tags = []
    for tag_element in tag_elements:
        tags.append(tag_element.text)
        
quotes.append(
    {
        'text': text,
        'author': author,
        'tags': ', '.join(tags) # merge the tags into a "A, B, ..., Z" string
    }
)

# reading  the "quotes.csv" file and creating it
# if not present
csv_file = open('quotes.csv', 'w', encoding='utf-8', newline='')

# initializing the writer object to insert data
# in the CSV file
writer = csv.writer(csv_file)

# writing the header of the CSV file
writer.writerow(['Text', 'Author', 'Tags'])

# writing each row of the CSV
for quote in quotes:
    writer.writerow(quote.values())

# terminating the operation and releasing the resources
csv_file.close()