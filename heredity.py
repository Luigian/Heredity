import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        0: 0.96,
        1: 0.03,
        2: 0.01
    },

    "trait": {

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },
        
        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        }
    },

    # Mutation probability
    "mutation": 0.01
}

PASS = {

    # Probability of passing a gene to a child given having no gene
    0: {
        True: PROBS["mutation"],
        False: 1 - PROBS["mutation"]
    },

    # Probability of passing a gene to a child given having one copy of gene
    1: {
        True: 0.5,
        False: 0.5
    },

    # Probability of passing a gene to a child given having two copies of gene
    2: {
        True: 1 - PROBS["mutation"],
        False: PROBS["mutation"]
    }
}

# Conditional probabilities of child genes given parents genes
"""
CHILD = {
     
    0: {
        0: {0: 0.9801, 1: 0.0198, 2: 0.0001},
        1: {0: 0.4950, 1: 0.5000, 2: 0.0050},
        2: {0: 0.0099, 1: 0.9802, 2: 0.0099}
    },

    1: {
        0: {0: 0.495, 1: 0.5, 2: 0.005},
        1: {0: 0.250, 1: 0.5, 2: 0.250},
        2: {0: 0.005, 1: 0.5, 2: 0.495}
    },

    2: {
        0: {0: 0.0099, 1: 0.9802, 2: 0.0099},
        1: {0: 0.0050, 1: 0.5000, 2: 0.4950},
        2: {0: 0.0001, 1: 0.0198, 2: 0.9801}
    }
}
"""


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    print("people")
    print(people)
    print("")

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
    print("probabilities")
    print(probabilities)
    print("")

    # Loop over all sets of people who might have the trait
    names = set(people)
    print("names")
    print(names)
    print("")
    print("power names")
    print(powerset(names))
    print("")

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
            # print(powerset(names - one_gene))
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                """
                have_trait: {'James'}
                one_gene: {'Harry'}
                two_genes: {'James'}
                zero : Lily
                no trait: Lily, Harry
                """
                # if "James" in have_trait and "Harry" in one_gene and "James" in two_genes and len(have_trait) == 1 and len(one_gene) == 1 and len(two_genes) == 1:
                
                    # print(f"have_trait: {have_trait}")
                    # print(f"one_gene: {one_gene}")
                    # print(f"two_genes: {two_genes}")

                p = joint_probability(people, one_gene, two_genes, have_trait)
                    # print(p)
                    # print("")
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

    p = 1.0
    names = set(people)

    for person in one_gene:
        if people[person]["mother"] == None and people[person]["father"] == None:
            p *= PROBS["gene"][1]
        else:
            mother = people[person]["mother"] 
            father = people[person]["father"] 
            mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
            from_mother = PASS[mother_genes][True] * PASS[father_genes][False] 
            from_father = PASS[father_genes][True] * PASS[mother_genes][False] 
            p *= from_mother + from_father
    
    for person in two_genes:
        if people[person]["mother"] == None and people[person]["father"] == None:
            p *= PROBS["gene"][2]
        else:
            mother = people[person]["mother"] 
            father = people[person]["father"] 
            mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
            p *= PASS[mother_genes][True] * PASS[father_genes][True]
    
    for person in names:
        if person not in one_gene and person not in two_genes:
            if people[person]["mother"] == None and people[person]["father"] == None:
                p *= PROBS["gene"][0]
            else:
                mother = people[person]["mother"] 
                father = people[person]["father"] 
                mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
                father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
                p *= PASS[mother_genes][False] * PASS[father_genes][False]

    for person in have_trait:
        person_genes = 1 if person in one_gene else 2 if person in two_genes else 0
        p *= PROBS["trait"][person_genes][True]
          
    for person in names:
        if person not in have_trait:
            person_genes = 1 if person in one_gene else 2 if person in two_genes else 0
            p *= PROBS["trait"][person_genes][False]

    return(p)


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    for person in probabilities:
        person_genes = 1 if person in one_gene else 2 if person in two_genes else 0
        probabilities[person]["gene"][person_genes] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        total_gene = probabilities[person]["gene"][0] + probabilities[person]["gene"][1] + probabilities[person]["gene"][2]
        probabilities[person]["gene"][0] /= total_gene
        probabilities[person]["gene"][1] /= total_gene
        probabilities[person]["gene"][2] /= total_gene
        total_trait = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        probabilities[person]["trait"][True] /= total_trait
        probabilities[person]["trait"][False] /= total_trait


if __name__ == "__main__":
    main()
