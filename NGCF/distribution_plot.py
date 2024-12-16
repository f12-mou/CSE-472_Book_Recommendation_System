import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the file
interaction_file = '../Data/amazon-book/train.txt'

# Parse the interaction data
user_interactions = []
book_interactions = {}

with open(interaction_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        user_id = int(parts[0])  # First value is the user ID
        book_ids = list(map(int, parts[1:]))  # Remaining values are book IDs
        
        # Add the interaction count for the user
        user_interactions.append(len(book_ids))
        
        # Update interaction counts for each book
        for book_id in book_ids:
            if book_id not in book_interactions:
                book_interactions[book_id] = 0
            book_interactions[book_id] += 1

# Convert book interactions to a list
book_interaction_counts = list(book_interactions.values())

# 1. Number of Users vs Number of Interactions
plt.figure(figsize=(6, 4))
plt.hist(user_interactions, bins=100, color='green', edgecolor='black')
plt.title('Number of Users vs Number of Interactions')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Users')
plt.xlim(0, 3000)  # Set x-axis limit
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("plot_1.png")
plt.show()

# 2. Number of Books vs Number of Interactions
plt.figure(figsize=(6, 4))
plt.hist(book_interaction_counts, bins=100, color='olive', edgecolor='black')
plt.title('Number of Books vs Number of Interactions')
plt.xlabel('Number of Interactions')
plt.ylabel('Number of Books')
plt.xlim(0, 800)  # Set x-axis limit
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("plot_2.png")
plt.show()

# 3. Violin Plot for User Interaction Distribution
plt.figure(figsize=(6, 4))
sns.violinplot(data=user_interactions, color='green')
plt.title('Distribution of Interactions per User')
plt.ylabel('Number of Interactions')
plt.savefig("plot_3.png")
plt.show()

# 4. Violin Plot for Book Interaction Distribution
plt.figure(figsize=(6, 4))
sns.violinplot(data=book_interaction_counts, color='olive')
plt.title('Distribution of Interactions per Book')
plt.ylabel('Number of Interactions')
plt.savefig("plot_4.png")
plt.show()
