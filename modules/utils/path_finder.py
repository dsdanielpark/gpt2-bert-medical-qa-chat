import os


def find_path(top_tree_path:str, type: str, include_word:str) -> list:
    """Find the path of a file or folder.

    :param top_tree_path: Enter the path to the top-level folder from which to start browsing files.
    :type top_tree_path: str
    :param type: Type to browse (folder or file)
    :type type: str
    :param include_word: Words included in what to search for (regardless of extension)
    :type include_word: str
    :return: List of files or folders browsed
    :rtype: list
    """
    pathes = []
    if top_tree_path == None:
        top_tree_path = os.getcwd()

    if type=='file':
        for (root, directories, files) in os.walk(top_tree_path):
            for file in files:
                if include_word in file:
                    file_path = os.path.join(root, file)
                    pathes.append(file_path)

    elif type=='folder':
        for (root, directories, files) in os.walk(top_tree_path):
            for dir in directories:
                if include_word in dir:
                    dir_path = os.path.join(root, dir)
                    pathes.append(dir_path)

    return pathes
