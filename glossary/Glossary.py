import json
import pprint
import textacy
import pandas as pd


class Glossary(dict):

    def __init__(self, dictionary=None):
        """Create glossary instance using existing dictionary or json file"""

        if dictionary is None:
            dictionary = {}
        # if we have json file
        if not isinstance(dictionary, dict):
            with open(dictionary) as json_data:
                dictionary = json.load(json_data)

        super(Glossary, self).__init__(dictionary)
        self.tags = set()
        self.tags_num = {}
        self._create_tags(dictionary)

    def __setitem__(self, key, value):
        key = self._normalize(key)
        super(Glossary, self).__setitem__(key, value)
        self._add_tag(key)

    def __delitem__(self, key):
        temp = Glossary()
        temp._create_tags(key)

        for key, value in temp.items():
            self.tags_num[key] -= value
            if self.tags_num[key] == 0:
                del self.tags_num[key]
                self.tags.remove(key)
        super(Glossary, self).__delitem__(key)

    def _add_tag(self, tag):
        """Add tag into instance dictionary"""
        tag = tag.lower().lstrip().rstrip()

        if tag not in self.tags:
            self.tags_num[tag] = 1
            self.tags.add(tag)
        else:
            self.tags_num[tag] += 1

    def _create_tags(self, dictionary):
        """Create dictionary with tags and their count"""

        for key, value in dictionary.items():
            self._add_tag(key)
            if value:
                self._create_tags(value)

    def get_path(self, tag):
        """Function that return path to the tag"""

        def _get(tag_r, dictionary):
            """To make function recursive"""

            for key, value in dictionary.items():
                if tag_r == key:
                    path.append(tag_r)
                    temp_path = "".join(map(lambda x: '["' + x + '"]', path))
                    paths.append(temp_path)
                    path.pop()
                elif not value:
                    continue
                else:
                    path.append(key)
                    _get(tag_r, value)
                    if path:
                        path.pop()

        paths = []
        path = []

        _get(tag, self)

        return paths

    @staticmethod
    def _format_path(paths):
        """Format path to appropriate list"""

        formatted_paths = []

        for path in paths:
            path = path.split('"')
            formatted_path = [p for p in path if not (p.startswith('[') or p.startswith(']'))]
            formatted_paths.append(formatted_path)

        return formatted_paths

    def get_parent(self, tag):
        """Return parent of tag if there are more than 1 -> return array of parents"""

        parents = []

        paths = self.get_path(tag)
        paths = self._format_path(paths)

        for path in paths:
            if len(path) < 2:
                parents.append([])
            else:
                parents.append(path[-2])

        return parents

    def get_children(self, tag):
        """Return tag's children"""

        children = []

        tags = self.search(tag)
        for tag in tags:
            child = list(tag.values())[0]
            child = list(child.keys())
            children.append(child)

        return children

    def search(self, tag, paths=None):
        """Get element of the glossary"""

        elements = []

        if tag not in self.tags:
            print('This tag is not part of glossary')
            return elements

        if paths:
            for path in paths:
                element = eval('{"' + tag + '" self{}'.format(path) + '}')
                elements.append(element)
        else:
            paths = self.get_path(tag)
            for path in paths:
                element = eval('{"' + tag + '": self{}'.format(path) + '}')
                elements.append(element)

        return elements

    def to_json(self):
        """Return json string"""

        json_string = json.dumps(self, indent=2)

        return json_string

    def save_as_json(self, file_name):
        """Save file as json with specific name"""

        name = file_name + '.json'

        with open(name, 'w') as fp:
            json.dump(self, fp, indent=2)

    @staticmethod
    def _normalize(tag):
        """Get normalized tag"""

        tag = textacy.spacier.utils.make_doc_from_text_chunks(tag, 'en')
        tag = textacy.extract.words(tag)
        return ' '.join(list(map(lambda x: x.lemma_, tag)))

    def _tags_to_df(self, normalize=True):
        """Create df of id-tag_name"""

        tags = ['None']
        tags.extend(sorted(self.tags))

        if normalize:
            tags = sorted(list(set(map(self._normalize, tags))))

        return pd.DataFrame(tags, columns=['tag'])

    def _links_to_df(self, normalize=True):
        """Create df of id-parent_id-child-id"""

        # Create dictionary for future dataframe
        links = dict(tag_id=[], parent_id=[], child_id=[])

        # Get ids and ids for
        ids = self._tags_to_df(normalize)

        for tag in sorted(self.tags):

            # Get parents and children
            parents = self.get_parent(tag)
            children = self.get_children(tag)

            # Get normalized version of parents and children
            if normalize:
                tag = self._normalize(tag)
                parents = list(map(lambda x: self._normalize(x) if x else x, parents))
                children = list(map(lambda x: list(map(lambda y: self._normalize(y), x)), children))

            # For each tag we have one parent and several children
            for parent, childs in zip(parents, children):
                # Get number of children to create appropriate number of rows
                num = len(childs) + (len(childs) == 0)

                # If we do not have parent assign it to 'None'
                if not parent:
                    parent = '' if normalize else 'None'

                # Get tag and parent ids
                tag_id = ids[ids.tag == tag].index[0]
                parent_id = ids[ids.tag == parent].index[0]

                childs_id = []
                for child in childs:
                    # If we do not have child -> assign to 'None'
                    if not child:
                        child = '' if normalize else 'None'

                    child_id = ids[ids.tag == child].index[0]
                    childs_id.append(child_id)

                # If we do not have any child -> assign to ['None']
                if not childs_id:
                    childs_id = [''] if normalize else ['None']

                # Append appropriate number of tag, parent ids w.r.t number of children
                links['tag_id'].extend([tag_id] * num)
                links['parent_id'].extend([parent_id] * num)
                links['child_id'].extend(childs_id)

        return pd.DataFrame(links)

    def tags_to_csv(self, path_or_buf=None, sep=',', normalize=True):
        """Create csv file of tags dataframe"""

        df = self._tags_to_df(normalize)
        df.to_csv(path_or_buf=path_or_buf, sep=sep)

    def links_to_csv(self, path_or_buf=None, sep=',', normalize=True):
        """Create csv file of links dataframe"""

        df = self._links_to_df(normalize)
        df.to_csv(path_or_buf=path_or_buf, sep=sep)

    def pretty_print(self, indent=1):
        for key, value in self.items():
            print('--|' * indent + str(key))
            if isinstance(value, dict):
                value = Glossary(value)
                value.pretty_print(indent + 1)
            else:
                print('--|' * (indent + 1) + str(value))

    def pprint(self, indent=1, width=80, depth=None, stream=None, compact=False):
        """Pretty print"""

        if stream:
            with open(stream, 'w') as out:
                pp = pprint.PrettyPrinter(indent, width, depth, out, compact=compact)
                pp.pprint(self)
        else:
            pp = pprint.PrettyPrinter(indent, width, depth, compact=compact)
            pp.pprint(self)


if __name__ == '__main__':
    from glossary.glossaries.pretty_glossary import pretty_core

    glossary = Glossary(pretty_core)