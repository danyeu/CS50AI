import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # If page has no outgoing links, all pages are equally likely
    if len(corpus[page]) == 0:
        return {pg: 1 / len(corpus) for pg in corpus}

    output = {}
    for pg in corpus:
        output[pg] = 0
        # visited randomly
        output[pg] += (1 - damping_factor) / len(corpus)
        # visited via link
        if pg in corpus[page]:
            output[pg] += damping_factor / len(corpus[page])

    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_size = n
    pages = list(corpus.keys())
    sample_count = {page: 0 for page in pages}

    # starting page
    current_page = random.choice(pages)
    sample_count[current_page] += 1
    n -= 1

    # sampling
    while n > 0:
        tm = transition_model(corpus, current_page, damping_factor)
        # randomly choose new page with probabilities from tm
        current_page = random.choices(pages, [tm[page] for page in pages])[0]
        sample_count[current_page] += 1
        n -= 1

    return {page: sample_count[page] / sample_size for page in pages}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    pr = {page: 1 / len(corpus) for page in pages}

    if damping_factor == 0:
        return pr

    # "A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself)."
    for page in pages:
        if len(corpus[page]) == 0:
            corpus[page] = {p for p in pages}

    # key:page, value: set of all pages that link to page
    links_to = {page: set() for page in pages}
    for page in pages:
        for p in corpus:
            if page in corpus[p]:
                links_to[page].add(p)

    # iteration
    loop = True
    while loop:
        loop = False
        for page in pages:
            sigma = 0
            for feeder in links_to[page]:
                sigma += pr[feeder] / len(corpus[feeder])
            old_pr = pr[page]
            pr[page] = ((1 - damping_factor) / len(corpus)) + damping_factor * sigma
            # continue looping only if change exceeds threshold for any page
            if abs(old_pr - pr[page]) > 0.001:
                loop = True

    return pr


if __name__ == "__main__":
    main()
