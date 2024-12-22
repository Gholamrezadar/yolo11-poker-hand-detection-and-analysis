from ultralytics import YOLO

def detect_cards(image_path, weights_path, conf=0.5):
    '''
    Detects cards in an image using YOLO11 model and returns the unique cards.

    Args:
        image_path (str): Path to the image file.
        weights_path (str): Path to the YOLO11 weights file.
        conf (float): Confidence threshold for the detection. Default is 0.5.

    Returns:
        list: List of unique cards detected in the image with confidence above the threshold sorted by their left position.
    '''

    model = YOLO(weights_path, task='detect')
    result = model.predict(image_path)[0]
    cards = [] # a list of tuples (left, card_name)
    cards_names = [] # a list of card names for deduplication
    summary = result.summary()

    for card in summary:
        if card['confidence'] >= conf:
            card_name = card['name']
            if card_name not in cards_names:
                cards_names.append(card_name)
                card_left = min(card['box']['x1'], card['box']['x2'])
                cards.append((card_left, card_name))
        
    # Sort the cards by their left position
    cards.sort(key=lambda x: x[0])

    # Extract the card names from the sorted list
    card_names = [card[1] for card in cards]

    return card_names 


def decode_cards(cards):
    '''
    Decodes the detected cards into a human-readable format.

    Args:
        cards (list): List of detected cards in the dataset format. e.g. 3C = "3 of Clubs".
    
    Returns:
        str: Human-readable format of the detected cards.
    '''
    
    card_names = {
        '2C': '2 of Clubs',
        '3C': '3 of Clubs',
        '4C': '4 of Clubs',
        '5C': '5 of Clubs',
        '6C': '6 of Clubs',
        '7C': '7 of Clubs',
        '8C': '8 of Clubs',
        '9C': '9 of Clubs',
        '10C': '10 of Clubs',
        'JC': 'Jack of Clubs',
        'QC': 'Queen of Clubs',
        'KC': 'King of Clubs',
        '2D': '2 of Diamonds',
        '3D': '3 of Diamonds',
        '4D': '4 of Diamonds',
        '5D': '5 of Diamonds',
        '6D': '6 of Diamonds',
        '7D': '7 of Diamonds',
        '8D': '8 of Diamonds',
        '9D': '9 of Diamonds',
        '10D': '10 of Diamonds',
        'JD': 'Jack of Diamonds',
        'QD': 'Queen of Diamonds',
        'KD': 'King of Diamonds',
        '2H': '2 of Hearts',
        '3H': '3 of Hearts',
        '4H': '4 of Hearts',
        '5H': '5 of Hearts',
        '6H': '6 of Hearts',
        '7H': '7 of Hearts',
        '8H': '8 of Hearts',
        '9H': '9 of Hearts',
        '10H': '10 of Hearts',
        'JH': 'Jack of Hearts',
        'QH': 'Queen of Hearts',
        'KH': 'King of Hearts',
        '2S': '2 of Spades',
        '3S': '3 of Spades',
        '4S': '4 of Spades',
        '5S': '5 of Spades',
        '6S': '6 of Spades',
        '7S': '7 of Spades',
        '8S': '8 of Spades',
        '9S': '9 of Spades',
        '10S': '10 of Spades',
        'JS': 'Jack of Spades',
        'QS': 'Queen of Spades',
        'KS': 'King of Spades',
        '2H': '2 of Hearts',
        '3H': '3 of Hearts',
        '4H': '4 of Hearts',
        '5H': '5 of Hearts',
        '6H': '6 of Hearts',
        '7H': '7 of Hearts',
        '8H': '8 of Hearts',
        '9H': '9 of Hearts',
        '10H': '10 of Hearts',
        'JH': 'Jack of Hearts',
        'QH': 'Queen of Hearts',
        'KH': 'King of Hearts',
        '2S': '2 of Spades',
        '3S': '3 of Spades',
        '4S': '4 of Spades',
        '5S': '5 of Spades',
        '6S': '6 of Spades',
        '7S': '7 of Spades',
        '8S': '8 of Spades',
        '9S': '9 of Spades',
        '10S': '10 of Spades',
        'JS': 'Jack of Spades',
        'QS': 'Queen of Spades',
        'KS': 'King of Spades',
    }
    
    return [card_names[card] for card in cards]

if __name__ == '__main__':
    # Test on the first image
    image_path = 'images/test_img_1.png'
    weights_path = 'weights/poker_best.pt'
    cards = detect_cards(image_path, weights_path)
    print("\n".join(decode_cards(cards)))

    # Test on the second image
    image_path = 'images/test_img_2.png'
    weights_path = 'weights/poker_best.pt'
    cards = detect_cards(image_path, weights_path)
    print("\n".join(decode_cards(cards)))

    # Test on the third image
    image_path = 'images/test_img_3.png'
    weights_path = 'weights/poker_best.pt'
    cards = detect_cards(image_path, weights_path)
    print("\n".join(decode_cards(cards)))