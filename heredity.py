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

    Conditional probabilities of child genes given parents genes:
    (these values are being calculated inside this function by doing join 
     probability using the values of the "passing" distribution)
    
    mother  | father    |   0       |   1       |   2
    0       |   0       |   0.9801  |   0.0198  |   0.0001
    0       |   1       |   0.4950  |   0.5000  |   0.0050
    0       |   2       |   0.0099  |   0.9802  |   0.0099
    1       |   0       |   0.495   |   0.5     |   0.005
    1       |   1       |   0.250   |   0.5     |   0.250
    1       |   2       |   0.005   |   0.5     |   0.495
    2       |   0       |   0.0099  |   0.9802  |   0.0099
    2       |   1       |   0.0050  |   0.5000  |   0.4950
    2       |   2       |   0.0001  |   0.0198  |   0.9801
    
    """

    # Probability of passing one gene to a child given no. of having copies
    passing = {
        0: {
            True: PROBS["mutation"],
            False: 1 - PROBS["mutation"]
        },
        1: {
            True: 0.5,
            False: 0.5
        },
        2: {
            True: 1 - PROBS["mutation"],
            False: PROBS["mutation"]
        }
    }

    names = set(people)
    p = 1.0

    # Joining the probability that everyone in set `one_gene` has one copy of the gene
    for person in one_gene:
        if people[person]["mother"] == None and people[person]["father"] == None:
            p *= PROBS["gene"][1]
        else:
            mother = people[person]["mother"] 
            father = people[person]["father"] 
            mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
            # Adding the two scenarios where a child gets one copy of the gene
            from_mother = passing[mother_genes][True] * passing[father_genes][False] 
            from_father = passing[father_genes][True] * passing[mother_genes][False] 
            p *= from_mother + from_father
    
    # Joinning the probability that everyone in set `two_genes` has two copies of the gene
    for person in two_genes:
        if people[person]["mother"] == None and people[person]["father"] == None:
            p *= PROBS["gene"][2]
        else:
            mother = people[person]["mother"] 
            father = people[person]["father"] 
            mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
            father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
            # The only scenario where a child gets two copies of the gene
            p *= passing[mother_genes][True] * passing[father_genes][True]
    
    # Joining the probability that everyone not in `one_gene` or `two_gene` does not have the gene
    for person in names:
        if person not in one_gene and person not in two_genes:
            if people[person]["mother"] == None and people[person]["father"] == None:
                p *= PROBS["gene"][0]
            else:
                mother = people[person]["mother"] 
                father = people[person]["father"] 
                mother_genes = 1 if mother in one_gene else 2 if mother in two_genes else 0
                father_genes = 1 if father in one_gene else 2 if father in two_genes else 0
                # The only scenario where a child gets zero copies of the gene
                p *= passing[mother_genes][False] * passing[father_genes][False]

    # Joining the probability that everyone in set `have_trait` has the trait
    for person in have_trait:
        person_genes = 1 if person in one_gene else 2 if person in two_genes else 0
        p *= PROBS["trait"][person_genes][True]
          
    # Joining the probability that everyone not in set` have_trait` does not have the trait
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
