class TrieNode:
    def __init__(self):
        self.children = {}  
        self.is_end_of_word = False
        
        # --- Atribut khusus untuk Search Engine ---
        self.term_id = None      
        self.doc_freq = 0         
        self.postings_offset = -1 

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.current_term_id = 0 

    def insert(self, word):
        """
        Memasukkan kata ke dalam Trie.
        Mengembalikan term_id dari kata tersebut (baru atau sudah ada).
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            node.is_end_of_word = True
            node.term_id = self.current_term_id
            self.current_term_id += 1
            
        node.doc_freq += 1
        return node.term_id

    def search(self, word):
        """
        Mencari kata di Trie.
        Mengembalikan objek TrieNode jika ketemu, atau None jika tidak ada.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
            
        if node.is_end_of_word:
            return node 
        return None