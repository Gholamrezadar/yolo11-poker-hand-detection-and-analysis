import itertools
from collections import Counter

# Define card ranks and suits
RANKS = "23456789TJQKA"
SUITS = "CDHS"

# Function to generate a deck
def generate_deck():
    return [rank + suit for rank in RANKS for suit in SUITS]

# Function to determine poker hand ranking (simplified rules)
def evaluate_hand(cards):
    ranks = sorted([card[0] for card in cards], key=RANKS.index)
    suits = [card[1] for card in cards]

    # How many times each rank appears 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
    rank_counts = Counter(ranks)
    sorted_rank_freqs = sorted(rank_counts.values(), reverse=True) # Sums to 7
    
    # How many times each suit appears ♦, ♥, ♠, ♣
    suit_counts = Counter(suits)

    is_flush = max(suit_counts.values()) >= 5
    is_straight = any(
        set(RANKS[i:i+5]).issubset(ranks) for i in range(len(RANKS) - 4)
    )

    wins = []
    
    # check for a straight flush
    if is_straight and is_flush:
        wins.append("Straight Flush")
    
    # Check for a flush
    if is_flush:
        wins.append("Flush")
    
    # Check for a straight
    if is_straight:
        wins.append("Straight")
    
    # Check for a one pair
    if max(rank_counts.values()) >= 2:
        wins.append("One Pair")

    # Check for a two pair
    if (sorted_rank_freqs[0] >= 2 and sorted_rank_freqs[1] >= 2) or sorted_rank_freqs[0] >= 4:
        wins.append("Two Pair")
    
    if max(rank_counts.values()) >= 3:
        wins.append("Three of a Kind")

    if max(rank_counts.values()) >= 4:
        wins.append("Four of a Kind")
    
    if sorted_rank_freqs[0] >= 3 and sorted_rank_freqs[1] >= 2:
        wins.append("Full House") 
    
    return wins

# Analyze probabilities
def analyze_hand(table_cards, hand_cards):
    '''Analyze the probabilities of winning poker hands.

    Args:
        table_cards (list): List of cards on the table (at most of length 5). e.g. ['2C', '7H', '9D'] 
        player_cards (list): List of cards on the player's hand (always of length 2). e.g. ['KS', '7D]
    
    Returns:
        dict: Dictionary of probabilities of winning poker hands.
    '''
    # Generate deck and remove known cards
    deck = generate_deck()
    known_cards = set(table_cards + hand_cards)
    remaining_deck = [card for card in deck if card not in known_cards]

    # Combine table cards and hand cards
    all_known_cards = table_cards + hand_cards

    # Simulate possible outcomes
    outcomes = Counter()
    total_cases = 0

    # Iterate over all possible combinations of remaining cards
    for extra_cards in itertools.combinations(remaining_deck, 5 - len(table_cards)):
        full_hand = all_known_cards + list(extra_cards)
        wins = evaluate_hand(full_hand)
        for win in wins:
            outcomes[win] += 1
        total_cases += 1

    # Convert counts to probabilities
    probabilities = {rank: count / total_cases for rank, count in outcomes.items()}

    # Sort probabilities by value in descending order
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
    return sorted_probabilities
