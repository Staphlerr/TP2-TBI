class TrieNode:
    def __init__(self):
        self.children = {}  
        self.is_end_of_word = False
        self.term_id = None      

class TrieIdMap:
    """
    Pengganti IdMap menggunakan struktur data Trie untuk menyimpan
    term-term dictionary. Memenuhi syarat Bonus Opsi 2 (TBI).
    """
    def __init__(self):
        self.root = TrieNode()
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term yang disimpan di Trie."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Mencari id dengan menyusuri (traversing) Trie.
        Kompleksitas O(m) di mana m adalah panjang karakter string.
        """
        node = self.root
        for char in s:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.term_id = len(self.id_to_str)
            self.id_to_str.append(s)
            
        return node.term_id

    def __getitem__(self, key):
        """Special method agar bisa diakses menggunakan kurung siku [...]"""
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

    def __contains__(self, key):
        """
        Mengecek apakah suatu kata ada di dalam Trie.
        Sangat efisien karena hanya menyusuri cabang yang diperlukan.
        """
        if type(key) is str:
            node = self.root
            for char in key:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end_of_word
        return False
        
    @property
    def str_to_id(self):
        """
        Trick/Wrapper: Karena di bsbi.py banyak kode yang memanggil 
        'word in self.term_id_map.str_to_id', property ini akan mengembalikan 
        class ini sendiri sehingga akan memicu fungsi __contains__ di atas.
        """
        return self