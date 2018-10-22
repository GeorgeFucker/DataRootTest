from textacy import similarity

class Categorizer:

    def __init__(self, tags=None, keyphrases=None, links=None):
        """ Create instance of class with appropriate tags, their dictionaries etc """

        self.categories = set()

        if keyphrases:
            keyphrases = [keyphrase for keyphrase in keyphrases]
        self.keyphrases = keyphrases

        if tags:
            self.set_tags(tags)

        if links:
            self.set_links(links)

    def __call__(self):
        """ Call categorize method """

        self.match()
        self.categorize()

    def set_tags(self, tags):
        """ Create dictionaries with tag_name : id and id : tag_name and tags"""

        self.tag_to_id = {}
        self.id_to_tag = {}
        self.tags = set()

        for id, tag in tags:
                self.tag_to_id[tag] = id
                self.id_to_tag[id] = tag
                self.tags.add(tag)

    def set_links(self, links):
        """ Create dictionaries parent_id : tag_id"""

        self.parent = {}

        for tag_id, parent_id, _ in links:
            if tag_id not in self.parent:
                self.parent[tag_id] = set()
            self.parent[tag_id].add(parent_id)

    def distance(self, str1, str2, distance='hamming', **kwargs):
        """ Measure the similar between two strings using """

        if isinstance(distance, str):
            distance = eval('similarity.{}'.format(distance))

        return distance(str1, str2, **kwargs)

    def match(self, distance='hamming', threshold=0.7, **kwargs):
        """ Match all tags that are similar
            **kwargs -> parameters for similarity function
        """

        self.matches = []

        for keyphrase in self.keyphrases:
            for tag in self.tags:
                if not tag:
                    continue
                d = self.distance(keyphrase[0], tag, distance=distance, **kwargs)
                if d > threshold:
                    self.matches.append((tag, d))


    def categorize(self, depth=2, macro_ctg=None, micro_ctg=None):
        """ Find all parents and create possible categories """

        start_depth = depth

        def get_parents(tag):
            """ Find 'n' parent of the tag"""

            nonlocal depth
            depth = depth - 1

            if depth == 0:
                return

            if isinstance(tag, str):
                tag = self.tag_to_id[tag]

            if tag:

                parents = self.parent[tag] - self.categories
                for parent in parents:
                    self.categories.add(parent)
                    get_parents(parent)

        matches = [self.tag_to_id[match] for match, _ in self.matches]
        self.categories.update(matches)

        for tag, _ in self.matches:
            get_parents(tag)
            depth = start_depth

        self.categories = {self.id_to_tag[category] for category in self.categories}

        if macro_ctg:
            self.macro_ctg = set(macro_ctg).intersection(self.categories)

        if micro_ctg:
            self.micro_ctg = set(micro_ctg).intersection(self.categories)