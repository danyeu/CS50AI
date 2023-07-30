import sys, itertools
from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.domains:
            words_to_remove = []
            for word in self.domains[v]:
                if len(word) != v.length:
                    words_to_remove.append(word)
            for word in words_to_remove:
                self.domains[v].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        if self.crossword.overlaps[x, y] is None:
            return False

        i, j = self.crossword.overlaps[x, y]

        remove_from_x = []
        ac = False
        for word_x in self.domains[x]:
            for word_y in self.domains[y]:
                if word_x[i] == word_y[j]:
                    if word_x == word_y:
                        continue
                    # arc consistent as at least one word (word_y) for this word_x does not conflict
                    ac = True
                    break
            # at least one possible word_y found, go to next word_x
            if ac:
                ac = False
                continue
            # if no word_y possible for word_x, word_x is to be removed from domains[x]
            remove_from_x.append(word_x)

        # if all word_x ok, stop here
        if len(remove_from_x) == 0:
            return False

        # else remove bad word_x from domains[x]
        for word in remove_from_x:
            self.domains[x].remove(word)
        return True

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # each arc is a tuple (x, y) of variables x != y

        if arcs is None:
            # enqueue all arcs
            arcs = [pair for pair in itertools.combinations(list(self.domains.keys()), 2)]

        while len(arcs) > 0:
            # dequeue an arc
            current_arc = arcs.pop(0)
            # check for ac
            if self.revise(current_arc[0], current_arc[1]):
                if len(self.domains[current_arc[0]]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(current_arc[0]):
                    if neighbor == current_arc[1]:
                        continue
                    arcs.append((neighbor, current_arc[0]))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for v in self.domains:
            if v not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        seen_words = set()
        for v in assignment:
            # word length
            if len(assignment[v]) != v.length:
                return False

            # unique words
            if assignment[v] in seen_words:
                return False
            seen_words.add(assignment[v])

        # overlaps
        for v in assignment:
            for neighbor in self.crossword.neighbors(v):
                if neighbor not in assignment:
                    continue
                if self.crossword.overlaps[v, neighbor] is None:
                    continue
                if assignment[v][self.crossword.overlaps[v, neighbor][0]] != assignment[neighbor][self.crossword.overlaps[v, neighbor][1]]:
                    return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        eliminations_per_word = {word: 0 for word in self.domains[var]}

        for word in self.domains[var]:
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    continue
                for neigbor_word in self.domains[neighbor]:
                    if word == neigbor_word:
                        eliminations_per_word[word] += 1
                        continue
                    if word[self.crossword.overlaps[var, neighbor][0]] != neigbor_word[self.crossword.overlaps[var, neighbor][1]]:
                        eliminations_per_word[word] += 1

        return sorted(eliminations_per_word, key=eliminations_per_word.get)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        remaining_vars = [var for var in self.domains if var not in assignment]

        # domain size and index in remaining_vars of var with the smallest domain size
        min_domain_size = len(self.domains[remaining_vars[0]])
        min_domain_indexes = [0]

        # finding the var in remaining_vars with the min_domain_size
        for v, var in enumerate(remaining_vars):
            if len(self.domains[var]) < min_domain_size:
                min_domain_size = len(self.domains[var])
                min_domain_indexes = [v]
            elif len(self.domains[var]) == min_domain_size:
                min_domain_indexes.append(v)

        # if only one var has the minimum domain size, return it
        if len(min_domain_indexes) == 1:
            return remaining_vars[min_domain_indexes[0]]

        # else, multiple vars have the minimum domain size (tie)
        # choose from these based on the variable with the largest degree (has the most neighbors)
        num_neighbors = [len(self.crossword.neighbors(remaining_vars[index])) for index in min_domain_indexes]
        max_neighbors_index = 0
        for i, num in enumerate(num_neighbors):
            if num > max_neighbors_index:
                # index in num_neighbors
                max_neighbors_index = i

        return remaining_vars[min_domain_indexes[max_neighbors_index]]


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # if assignment complete, return it
        if self.assignment_complete(assignment):
            return assignment

        current_var = self.select_unassigned_variable(assignment)
        for word in self.order_domain_values(current_var, assignment):
                assignment[current_var] = word
                if self.consistent(assignment):
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                assignment.pop(current_var)
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
