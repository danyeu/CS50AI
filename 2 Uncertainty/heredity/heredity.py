import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    want_to_find = {person: {"gene": 0, "trait": False} for person in people}
    for person in one_gene:
        want_to_find[person]["gene"] = 1
    for person in two_genes:
        want_to_find[person]["gene"] = 2
    for person in have_trait:
        want_to_find[person]["trait"] = True


    # returns prob of getting x genes from a single parent with y genes
    def p_get_x_given_parent_y(x: int, y: int):
        if x not in [0, 1] or y not in [0, 1, 2]:
            raise ValueError
        if y == 0:
            if x == 0:
                return 1 - PROBS["mutation"]
            else:
                return PROBS["mutation"]
        elif y == 1:
            return 0.5
        else:
            if x == 0:
                return PROBS["mutation"]
            else:
                return 1 - PROBS["mutation"]


    joint_p = 1
    for person in want_to_find:

        # P(trait|no trait) == P(no trait|trait) == 0
        if people[person]["trait"] is not None and want_to_find[person]["trait"] != people[person]["trait"]:
            return 0

        # === parents unknown ===
        if people[person]["mother"] is None and people[person]["father"] is None:
            # population prob that someone has the same trait status as this person
            p_persons_trait = sum([PROBS["gene"][i] * PROBS["trait"][i][want_to_find[person]["trait"]] for i in range(3)])

            if people[person]["trait"] is None:
                # parents unknown, trait unknown - the persons probability is just from the population table
                joint_p *= PROBS["trait"][want_to_find[person]["gene"]][want_to_find[person]["trait"]] * PROBS["gene"][want_to_find[person]["gene"]]
            else:
                # parents unknown, trait known - want P(#genes|trait_status) = P(trait_status|#genes) * P(#genes) / P(trait_status)
                joint_p *= PROBS["trait"][want_to_find[person]["gene"]][want_to_find[person]["trait"]] * PROBS["gene"][want_to_find[person]["gene"]] / p_persons_trait
            continue

        # === parents known ===
        p_has_x_genes = 1
        if want_to_find[person]["gene"] == 0:
            # child has 0 = 0 from mother + 0 from father
            for parent in ["mother", "father"]:
                if people[person][parent] is None:
                    p_has_x_genes *= (PROBS["gene"][0] * (1 - PROBS["mutation"])) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * PROBS["mutation"])
                else:
                    p_has_x_genes *= p_get_x_given_parent_y(0, want_to_find[people[person][parent]]["gene"])

        elif want_to_find[person]["gene"] == 1:
            # P(child has 1 with 0 from mother + 1 from father)
            p_m0f1 = 0
            if people[person]["mother"] is None:
                p_m0f1 = (PROBS["gene"][0] * (1 - PROBS["mutation"])) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * PROBS["mutation"])
            else:
                p_m0f1 = p_get_x_given_parent_y(0, want_to_find[people[person]["mother"]]["gene"])
            if people[person]["father"] is None:
                p_m0f1 *= (PROBS["gene"][0] * PROBS["mutation"]) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * (1 - PROBS["mutation"]))
            else:
                p_m0f1 *= p_get_x_given_parent_y(1, want_to_find[people[person]["father"]]["gene"])

            # P(child has 1 with 1 from mother + 0 from father)
            p_m1f0 = 0
            if people[person]["mother"] is None:
                p_m1f0 = (PROBS["gene"][0] * PROBS["mutation"]) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * (1 - PROBS["mutation"]))
            else:
                p_m1f0 = p_get_x_given_parent_y(1, want_to_find[people[person]["mother"]]["gene"])
            if people[person]["father"] is None:
                p_m1f0 *= (PROBS["gene"][0] * (1 - PROBS["mutation"])) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * PROBS["mutation"])
            else:
                p_m1f0 *= p_get_x_given_parent_y(0, want_to_find[people[person]["father"]]["gene"])

            # P(child has 1) = P(0 from mother + 1 from father) + P(1 from mother + 0 from father)
            p_has_x_genes = p_m0f1 + p_m1f0

        elif want_to_find[person]["gene"] == 2:
            # child has 2 = 1 from mother + 1 from father
            for parent in ["mother", "father"]:
                if people[person][parent] is None:
                    p_has_x_genes *= (PROBS["gene"][0] * PROBS["mutation"]) + (PROBS["gene"][1] * 0.5) + (PROBS["gene"][2] * (1 - PROBS["mutation"]))
                else:
                    p_has_x_genes *= p_get_x_given_parent_y(1, want_to_find[people[person][parent]]["gene"])

        # p_has_x_genes = P(has_x_genes|parents data)
        # P(has_x_genes, trait_state) = p_has_x_genes * P(trait_state|has_x_genes)
        joint_p *= p_has_x_genes * PROBS["trait"][want_to_find[person]["gene"]][want_to_find[person]["trait"]]

    return joint_p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        factor = 1 / sum([probabilities[person]["trait"][b] for b in [True, False]])
        for b in [True, False]:
            probabilities[person]["trait"][b] *= factor
        factor = 1 / sum([probabilities[person]["gene"][i] for i in range(3)])
        for i in range(3):
            probabilities[person]["gene"][i] *= factor


if __name__ == "__main__":
    main()
