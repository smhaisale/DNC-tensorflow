import tensorflow as tf
import numpy as np
import utility

class SparseMemory:

    def __init__(self, words_num=256, word_size=64, read_heads=4, batch_size=1, k=8):
        """
        constructs a memory matrix with read heads and a write head as described
        in the DNC paper
        http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

        Parameters:
        ----------
        words_num: int
            the maximum number of words that can be stored in the memory at the
            same time
        word_size: int
            the size of the individual word in the memory
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
        """

        self.words_num = words_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size
        self.K = k

        # a words_num x words_num identity matrix
        self.I = tf.constant(np.identity(words_num, dtype=np.float32))

        # maps the indices from the 2D array of free list per batch to
        # their corresponding values in the flat 1D array of ordered_allocation_weighting
        self.index_mapper = tf.constant(
            np.cumsum([0] + [words_num] * (batch_size - 1), dtype=np.int32)[:, np.newaxis]
        )

    def init_memory(self):
        """
        returns the initial values for the memory Parameters

        Returns: Tuple
        """

        return (
            tf.fill([self.batch_size, self.words_num, self.word_size], 1e-6),   # initial memory matrix
            tf.zeros([self.batch_size, self.words_num, ]),                      # initial usage vector
            tf.zeros([self.batch_size, self.words_num, ]),                      # initial precedence vector
            tf.zeros([self.batch_size, self.words_num, self.words_num]),        # initial forward_link matrix
            tf.zeros([self.batch_size, self.words_num, self.words_num]),        # initial backward_link matrix
            tf.fill([self.batch_size, self.words_num, ], 1e-6),                 # initial write weighting
            tf.fill([self.batch_size, self.words_num, self.read_heads], 1e-6),  # initial read weightings
            tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6),  # initial read vectors
        )

    def get_lookup_weighting(self, memory_matrix, keys, strengths):
        """
        retrieves a content-based addressing weighting given the keys

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix to lookup in
        keys: Tensor (batch_size, word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key

        Returns: Tensor (batch_size, words_num, number_of_keys)
            The list of lookup weightings for each provided key
        """

        normalized_memory = tf.nn.l2_normalize(memory_matrix, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)

        similarity = tf.matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)

        return tf.nn.softmax(similarity * strengths, 1)


    def get_interpolation_weighting(self, usage_vector, read_weightings, write_weighting, free_gates):
        """

        Args:
        	usage_vector:
        	read_weightings:
        	write_weighting:
        	free_gates:

        Returns:

        """
        retention_vector = tf.reduce_prod(1 - read_weightings * free_gates, 2)
        recently_used = 0.005 - (retention_vector + write_weighting)
        recently_used = tf.minimum(recently_used, 0)

        updated_usage = tf.where(tf.equal(recently_used, 0), tf.fill(tf.shape(usage_vector), 1e-16), usage_vector)

        least_recently_used = tf.reduce_max(updated_usage)
        interpolation_weighting = tf.where(tf.equal(least_recently_used, updated_usage), tf.ones(tf.shape(updated_usage)), tf.fill(tf.shape(updated_usage), 1e-16))

        return interpolation_weighting

    # Only used during write
    def update_usage_vector(self, usage_vector, read_weightings, write_weighting, free_gates):
        """
        updates and returns the usgae vector given the values of the free gates
        and the usage_vector, read_weightings, write_weighting from previous step

        Parameters:
        ----------
        usage_vector: Tensor (batch_size, words_num)
        read_weightings: Tensor (batch_size, words_num, read_heads)
        write_weighting: Tensor (batch_size, words_num)
        free_gates: Tensor (batch_size, read_heads, )

        Returns: Tensor (batch_size, words_num, )
            the updated usage vector
        """
        free_gates = tf.expand_dims(free_gates, 1)

        retention_vector = tf.reduce_prod(1 - read_weightings * free_gates, 2)
        updated_usage = (usage_vector + write_weighting - usage_vector * write_weighting)  * retention_vector

        return updated_usage

    # Only used during write
    # Needs to use a usage vector to compute and return a sparse allocation vector
    # Usage vector records, for each memory word, number of time steps since that memory word has been accessed
    def get_allocation_weighting(self, sorted_usage, free_list):
        """
        retrieves I(U,t) the writing allocation weighting based on the usage free list

        Parameters:
        ----------
        sorted_usage: Tensor (batch_size, words_num, )
            the usage vector sorted ascendingly
        free_list: Tensor (batch, words_num, )
            the original indices of the sorted usage vector

        Returns: Tensor (batch_size, words_num, )
            the allocation weighting for each word in memory
        """

        shifted_cumprod = tf.cumprod(sorted_usage, axis = 1, exclusive=True)
        unordered_allocation_weighting = (1 - sorted_usage) * shifted_cumprod

        mapped_free_list = free_list + self.index_mapper
        flat_unordered_allocation_weighting = tf.reshape(unordered_allocation_weighting, (-1,))
        flat_mapped_free_list = tf.reshape(mapped_free_list, (-1,))
        flat_container = tf.TensorArray(tf.float32, self.batch_size * self.words_num)

        flat_ordered_weightings = flat_container.scatter(
            flat_mapped_free_list,
            flat_unordered_allocation_weighting
        )

        packed_weightings = flat_ordered_weightings.stack()
        return tf.reshape(packed_weightings, (self.batch_size, self.words_num))


    def update_write_weighting(self, lookup_weighting, allocation_weighting, write_gate, allocation_gate):
        """
        updates and returns the current write_weighting

        Parameters:
        ----------
        lookup_weighting: Tensor (batch_size, words_num, 1)
            the weight of the lookup operation in writing
        allocation_weighting: Tensor (batch_size, words_num)
            the weight of the allocation operation in writing
        write_gate: (batch_size, 1)
            the fraction of writing to be done
        allocation_gate: (batch_size, 1)
            the fraction of allocation to be done

        Returns: Tensor (batch_size, words_num)
            the updated write_weighting
        """

        # remove the dimension of 1 from the lookup_weighting
        lookup_weighting = tf.squeeze(lookup_weighting)

        updated_write_weighting = write_gate * (allocation_gate * allocation_weighting + (1 - allocation_gate) * lookup_weighting)

        return updated_write_weighting


    def update_memory(self, memory_matrix, write_weighting, write_vector, erase_vector):
        """
        updates and returns the memory matrix given the weighting, write and erase vectors
        and the memory matrix from previous step

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix from previous step
        write_weighting: Tensor (batch_size, words_num)
            the weight of writing at each memory location
        write_vector: Tensor (batch_size, word_size)
            a vector specifying what to write
        erase_vector: Tensor (batch_size, word_size)
            a vector specifying what to erase from memory

        Returns: Tensor (batch_size, words_num, word_size)
            the updated memory matrix
        """

        # expand data with a dimension of 1 at multiplication-adjacent location
        # to force matmul to behave as an outer product
        write_weighting = tf.expand_dims(write_weighting, 2)
        write_vector = tf.expand_dims(write_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)

        erasing = memory_matrix * (1 - tf.matmul(write_weighting, erase_vector))
        writing = tf.matmul(write_weighting, write_vector)
        updated_memory = erasing + writing

        return updated_memory


    def update_precedence_vector(self, precedence_vector, write_weighting):
        """
        updates the precedence vector given the latest write weighting
        and the precedence_vector from last step

        Parameters:
        ----------
        precedence_vector: Tensor (batch_size. words_num)
            the precedence vector from the last time step
        write_weighting: Tensor (batch_size,words_num)
            the latest write weighting for the memory

        Returns: Tensor (batch_size, words_num)
            the updated precedence vector
        """

        reset_factor = 1 - tf.reduce_sum(write_weighting, 1, keep_dims=True)
        updated_precedence_vector = reset_factor * precedence_vector + write_weighting

        return updated_precedence_vector


    def update_link_matrices(self, precedence_vector, forward_link_matrix, backward_link_matrix, write_weighting):
        """
        updates and returns the temporal link matrix for the latest write
        given the precedence vector and the link matrix from previous step

        Parameters:
        ----------
        precedence_vector: Tensor (batch_size, words_num)
            the precedence vector from the last time step
        forward_link_matrix: Tensor (batch_size, words_num, K)
            the link matrix form the last step
        backward_link_matrix: Tensor (batch_size, words_num, K)
            the link matrix form the last step
        write_weighting: Tensor (batch_size, words_num)
            the latest write_weighting for the memory

        Returns: Tensor (batch_size, words_num, words_num)
            the updated temporal link matrix
        """

        write_weighting_i = tf.expand_dims(write_weighting, 2)
        precedence_vector_j = tf.expand_dims(precedence_vector, 1)

        # TODO: Need to use below for backward link matrix but gradients NaN out (why?)
        write_weighting_j = tf.expand_dims(write_weighting, 1)
        precedence_vector_i = tf.expand_dims(precedence_vector, 2)

        updated_forward_link_matrix = (1 - write_weighting_i) * forward_link_matrix + tf.matmul(write_weighting_i, precedence_vector_j)
        updated_backward_link_matrix = (1 - write_weighting_i) * backward_link_matrix + tf.matmul(write_weighting_i, precedence_vector_j)

        return updated_forward_link_matrix, updated_backward_link_matrix


    def get_directional_weightings(self, read_weightings, forward_link_matrix, backward_link_matrix):
        """
        computes and returns the forward and backward reading weightings
        given the read_weightings from the previous step

        Parameters:
        ----------
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read weightings from the last time step
        forward_link_matrix: Tensor (batch_size, words_num, K)
            the forward link matrix
        backward_link_matrix: Tensor (batch_size, words_num, K)
            the backward link matrix

        Returns: Tuple
            forward weighting: Tensor (batch_size, words_num, read_heads),
            backward weighting: Tensor (batch_size, words_num, read_heads)
        """

        forward_weighting = tf.matmul(forward_link_matrix, read_weightings)
        backward_weighting = tf.matmul(backward_link_matrix, read_weightings, adjoint_a=True)

        return forward_weighting, backward_weighting


    def update_read_weightings(self, lookup_weightings, forward_weighting, backward_weighting, read_mode):
        """
        updates and returns the current read_weightings

        Parameters:
        ----------
        lookup_weightings: Tensor (batch_size, words_num, read_heads)
            the content-based read weighting
        forward_weighting: Tensor (batch_size, words_num, read_heads)
            the forward direction read weighting
        backward_weighting: Tensor (batch_size, words_num, read_heads)
            the backward direction read weighting
        read_mode: Tesnor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes

        Returns: Tensor (batch_size, words_num, read_heads)
        """

        backward_mode = tf.expand_dims(read_mode[:, 0, :], 1) * backward_weighting
        lookup_mode = tf.expand_dims(read_mode[:, 1, :], 1) * lookup_weightings
        forward_mode = tf.expand_dims(read_mode[:, 2, :], 1) * forward_weighting
        updated_read_weightings = backward_mode + lookup_mode + forward_mode

        return updated_read_weightings


    def update_read_vectors(self, memory_matrix, read_weightings):
        """
        reads, updates, and returns the read vectors of the recently updated memory

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the recently updated memory matrix
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the amount of info to read from each memory location by each read head

        Returns: Tensor (word_size, read_heads)
        """

        updated_read_vectors = tf.matmul(memory_matrix, read_weightings, adjoint_a=True)

        return updated_read_vectors
    
    def write(self, memory_matrix, usage_vector, read_weightings, write_weighting,
              precedence_vector, forward_link_matrix, backward_link_matrix,  key, strength, free_gates,
              allocation_gate, write_gate, write_vector, erase_vector):
        """
        defines the complete pipeline of writing to memory gievn the write variables
        and the memory_matrix, usage_vector, link_matrix, and precedence_vector from
        previous step

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the memory matrix from previous step
        usage_vector: Tensor (batch_size, words_num)
            the usage_vector from the last time step
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read_weightings from the last time step
        write_weighting: Tensor (batch_size, words_num)
            the write_weighting from the last time step
        precedence_vector: Tensor (batch_size, words_num)
            the precedence vector from the last time step
        forward_link_matrix: Tensor (batch_size, words_num, K)
            the forward_link_matrix from previous step
        backward_link_matrix: Tensor (batch_size, words_num, K)
            the backward_link_matrix from previous step
        key: Tensor (batch_size, word_size, 1)
            the key to query the memory location with
        strength: (batch_size, 1)
            the strength of the query key
        free_gates: Tensor (batch_size, read_heads)
            the degree to which location at read haeds will be freed
        allocation_gate: (batch_size, 1)
            the fraction of writing that is being allocated in a new locatio
        write_gate: (batch_size, 1)
            the amount of information to be written to memory
        write_vector: Tensor (batch_size, word_size)
            specifications of what to write to memory
        erase_vector: Tensor(batch_size, word_size)
            specifications of what to erase from memory

        Returns : Tuple
            the updated usage vector: Tensor (batch_size, words_num)
            the updated write_weighting: Tensor(batch_size, words_num)
            the updated memory_matrix: Tensor (batch_size, words_num, words_size)
            the updated link matrix: Tensor(batch_size, words_num, words_num)
            the updated precedence vector: Tensor (batch_size, words_num)
        """

        lookup_weighting = self.get_lookup_weighting(memory_matrix, key, strength)
        new_usage_vector = self.update_usage_vector(usage_vector, read_weightings, write_weighting, free_gates)

        sorted_usage, free_list = tf.nn.top_k(-1 * new_usage_vector, self.words_num)
        sorted_usage = -1 * sorted_usage

        allocation_weighting = self.get_allocation_weighting(sorted_usage, free_list)
        interpolation_weighting = self.get_interpolation_weighting(usage_vector, read_weightings, write_weighting, free_gates)

        new_write_weighting = self.update_write_weighting(lookup_weighting, interpolation_weighting, write_gate, allocation_gate)
        new_memory_matrix = self.update_memory(memory_matrix, new_write_weighting, write_vector, erase_vector)
        new_forward_link_matrix, new_backward_link_matrix = self.update_link_matrices(precedence_vector, forward_link_matrix, backward_link_matrix, new_write_weighting)
        new_precedence_vector = self.update_precedence_vector(precedence_vector, new_write_weighting)

        return new_usage_vector, new_write_weighting, new_memory_matrix, new_forward_link_matrix, new_backward_link_matrix, new_precedence_vector


    def read(self, memory_matrix, read_weightings, keys, strengths, forward_link_matrix, backward_link_matrix, read_modes, k=8):
        """
        defines the complete pipeline for reading from memory

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, words_num, word_size)
            the updated memory matrix from the last writing
        read_weightings: Tensor (batch_size, words_num, read_heads)
            the read weightings form the last time step
        keys: Tensor (batch_size, word_size, read_heads)
            the kyes to query the memory locations with
        strengths: Tensor (batch_size, read_heads)
            the strength of each read key
        forward_link_matrix: Tensor (batch_size, words_num, K)
            the updated forward link matrix from the last writing
        backward_link_matrix: Tensor (batch_size, words_num, K)
            the updated backward link matrix from the last writing
        read_modes: Tensor (batch_size, 3, read_heads)
            the softmax distribution between the three read modes

        Returns: Tuple
            the updated read_weightings: Tensor(batch_size, words_num, read_heads)
            the recently read vectors: Tensor (batch_size, word_size, read_heads)
        """
        lookup_weighting = self.get_lookup_weighting(memory_matrix, keys, strengths)
        forward_weighting, backward_weighting = self.get_directional_weightings(read_weightings, forward_link_matrix, backward_link_matrix)
        new_read_weightings = self.update_read_weightings(lookup_weighting, forward_weighting, backward_weighting, read_modes)

        new_read_weightings = new_read_weightings[0]
        new_read_weightings = tf.transpose(new_read_weightings)
        values, indices = tf.nn.top_k(new_read_weightings, k=8)
        sparse_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
        sparse_range = tf.tile(sparse_range, [1, k])
        full_indices = tf.concat(axis=2, values=[tf.expand_dims(sparse_range, 2), tf.expand_dims(indices, 2)])
        full_indices = tf.reshape(full_indices, [-1, 2])
        sparse_new_read_weightings = tf.sparse_to_dense(full_indices, new_read_weightings.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
        sparse_new_read_weightings = tf.transpose(sparse_new_read_weightings)
        sparse_new_read_weightings = tf.expand_dims(sparse_new_read_weightings, 0)
        sparse_new_read_vectors = self.update_read_vectors(memory_matrix, sparse_new_read_weightings)
        return sparse_new_read_weightings, sparse_new_read_vectors
