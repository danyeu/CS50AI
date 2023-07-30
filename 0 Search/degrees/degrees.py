import csv
import sys

from util import Node, StackFrontier, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected.")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


"""=== My functions below ==="""

def shortest_path(source, target):
    """
    Returns the shortest list that connect the source to the target:
    [(movie_id, person_id), (movie_id, person_id), ...]

    The first movie_id should be the one that source is in.
    The last person_id should be the target.

    If no possible path, returns None.
    """

    # Initialize frontier to just the starting position
    # Starting node has no parent node, and no action taken to get to that node
    start = Node(state=source, parent=None, action=None)
    # State = person_id, parent = parent Node, action = movie_id to get to that state

    # Track the states explored
    explored_set = ExploredSet()

    # Track the nodes to explore: breadth-first search
    frontier = QueueFrontier()
    # Initialise frontier
    frontier.add(start)

    # Loop until no nodes left to check
    while not frontier.empty():
        # Set node to check and remove from frontier
        node = frontier.remove()
        # Set state as explored
        explored_set.add(node.state)

        # If we are in the goal state, return the solution
        if node.state == target:
            return node_history(node)

        # Else, add the possible child nodes to the frontier if their states are not already there/not explored already
            # Actions is a set of (movid_id, person_id) tuples that you can get to from the current state (person_id)
        actions = neighbors_for_person(node.state)
        # For each tuple in actions
        for mid_pid in actions:
            # If the target exists in a possible child node, immediately recognise the answer
            if mid_pid[1] == target:
                node = Node(state=target, parent=node, action=mid_pid[0])
                return node_history(node)
            # Add node to frontier if no explored and not already there
            if (not frontier.contains_state(mid_pid[1])) and (not explored_set.contains_state(mid_pid[1])):
                frontier.add(Node(state=mid_pid[1], parent=node, action=mid_pid[0]))

        continue

    # Here, frontier is empty and nothing has been returned yet, there must be no connection between the actors
    return None


# Set of all states (person_id) explored
class ExploredSet():
    def __init__(self):
        self.explored_set = set()

    def add(self, state):
        self.explored_set.add(state)

    def contains_state(self, state):
        if state in self.explored_set:
            return True
        return False


# Given a node, returns a list of tuples (movie_id, person_id) to get to that node
def node_history(node: Node):
    history = []

    # Loop through all parent nodes
    while True:
        # Return if no parent node (i.e. we are at the initial node)
        if node.parent is None:
            # Reverse the list to give correct order of exploration
            history.reverse()
            return history
        # Add tuple to list for each node, parent node, grandparent node, etc
        history.append((node.action, node.state))
        # Go up the family tree
        node = node.parent


if __name__ == "__main__":
    main()
