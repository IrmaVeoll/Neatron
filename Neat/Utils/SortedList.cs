using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Neat.Utils
{
    // ReSharper disable InconsistentNaming
    internal sealed class IDictionaryDebugView<K, V>
    {
        private readonly IDictionary<K, V> _dict;

        public IDictionaryDebugView(IDictionary<K, V> dictionary)
        {
            _dict = dictionary ?? throw new ArgumentNullException(nameof(dictionary));
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public KeyValuePair<K, V>[] Items
        {
            get
            {
                var items = new KeyValuePair<K, V>[_dict.Count];
                _dict.CopyTo(items, 0);
                return items;
            }
        }
    }

    internal sealed class DictionaryKeyCollectionDebugView<TKey>
    {
        private readonly ICollection<TKey> _collection;

        public DictionaryKeyCollectionDebugView(ICollection<TKey> collection)
        {
            _collection = collection ?? throw new ArgumentNullException(nameof(collection));
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public TKey[] Items
        {
            get
            {
                var items = new TKey[_collection.Count];
                _collection.CopyTo(items, 0);
                return items;
            }
        }
    }

    internal sealed class DictionaryValueCollectionDebugView<TValue>
    {
        private readonly ICollection<TValue> _collection;

        public DictionaryValueCollectionDebugView(ICollection<TValue> collection)
        {
            _collection = collection ?? throw new ArgumentNullException(nameof(collection));
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public TValue[] Items
        {
            get
            {
                var items = new TValue[_collection.Count];
                _collection.CopyTo(items, 0);
                return items;
            }
        }
    }
    
    // Licensed to the .NET Foundation under one or more agreements.
    // The .NET Foundation licenses this file to you under the MIT license.
    // See the LICENSE file in the project root for more information.

    // The SortedDictionary class implements a generic sorted list of keys 
    // and values. Entries in a sorted list are sorted by their keys and 
    // are accessible both by key and by index. The keys of a sorted dictionary
    // can be ordered either according to a specific IComparer 
    // implementation given when the sorted dictionary is instantiated, or 
    // according to the IComparable implementation provided by the keys 
    // themselves. In either case, a sorted dictionary does not allow entries
    // with duplicate or null keys.
    // 
    // A sorted list internally maintains two arrays that store the keys and
    // values of the entries. The capacity of a sorted list is the allocated
    // length of these internal arrays. As elements are added to a sorted list, the
    // capacity of the sorted list is automatically increased as required by
    // reallocating the internal arrays.  The capacity is never automatically 
    // decreased, but users can call either TrimExcess or 
    // Capacity explicitly.
    // 
    // The GetKeyList and GetValueList methods of a sorted list
    // provides access to the keys and values of the sorted list in the form of
    // List implementations. The List objects returned by these
    // methods are aliases for the underlying sorted list, so modifications
    // made to those lists are directly reflected in the sorted list, and vice
    // versa.
    // 
    // The SortedList class provides a convenient way to create a sorted
    // copy of another dictionary, such as a Hashtable. For example:
    // 
    // Hashtable h = new Hashtable();
    // h.Add(...);
    // h.Add(...);
    // ...
    // SortedList s = new SortedList(h);
    // 
    // The last line above creates a sorted list that contains a copy of the keys
    // and values stored in the hashtable. In this particular example, the keys
    // will be ordered according to the IComparable interface, which they
    // all must implement. To impose a different ordering, SortedList also
    // has a constructor that allows a specific IComparer implementation to
    // be specified.
    // 
    [DebuggerTypeProxy(typeof(IDictionaryDebugView<,>))]
    [DebuggerDisplay("Count = {Count}")]
    [Serializable]
    public class SortedList<TKey, TValue> :
        IDictionary<TKey, TValue>, IDictionary, IReadOnlyDictionary<TKey, TValue>
    {
        // ReSharper disable FieldCanBeMadeReadOnly.Local
        private TKey[] keys; // Do not rename (binary serialization)
        private TValue[] values; // Do not rename (binary serialization)
        private int _size; // Do not rename (binary serialization)
        private int version; // Do not rename (binary serialization)
        private IComparer<TKey> comparer; // Do not rename (binary serialization)
        private KeyList keyList; // Do not rename (binary serialization)
        private ValueList valueList; // Do not rename (binary serialization)

        private const int DefaultCapacity = 4;

        // Constructs a new sorted list. The sorted list is initially empty and has
        // a capacity of zero. Upon adding the first element to the sorted list the
        // capacity is increased to DefaultCapacity, and then increased in multiples of two as
        // required. The elements of the sorted list are ordered according to the
        // IComparable interface, which must be implemented by the keys of
        // all entries added to the sorted list.
        public SortedList()
        {
            keys = Array.Empty<TKey>();
            values = Array.Empty<TValue>();
            _size = 0;
            comparer = Comparer<TKey>.Default;
        }

        // Constructs a new sorted list. The sorted list is initially empty and has
        // a capacity of zero. Upon adding the first element to the sorted list the
        // capacity is increased to 16, and then increased in multiples of two as
        // required. The elements of the sorted list are ordered according to the
        // IComparable interface, which must be implemented by the keys of
        // all entries added to the sorted list.
        //
        public SortedList(int capacity)
        {
            if (capacity < 0)
                throw new ArgumentOutOfRangeException(nameof(capacity));
            keys = new TKey[capacity];
            values = new TValue[capacity];
            comparer = Comparer<TKey>.Default;
        }

        // Constructs a new sorted list with a given IComparer
        // implementation. The sorted list is initially empty and has a capacity of
        // zero. Upon adding the first element to the sorted list the capacity is
        // increased to 16, and then increased in multiples of two as required. The
        // elements of the sorted list are ordered according to the given
        // IComparer implementation. If comparer is null, the
        // elements are compared to each other using the IComparable
        // interface, which in that case must be implemented by the keys of all
        // entries added to the sorted list.
        // 
        public SortedList(IComparer<TKey> comparer)
            : this()
        {
            if (comparer != null)
            {
                this.comparer = comparer;
            }
        }

        // Constructs a new sorted dictionary with a given IComparer
        // implementation and a given initial capacity. The sorted list is
        // initially empty, but will have room for the given number of elements
        // before any reallocations are required. The elements of the sorted list
        // are ordered according to the given IComparer implementation. If
        // comparer is null, the elements are compared to each other using
        // the IComparable interface, which in that case must be implemented
        // by the keys of all entries added to the sorted list.
        // 
        public SortedList(int capacity, IComparer<TKey> comparer)
            : this(comparer)
        {
            Capacity = capacity;
        }

        // Constructs a new sorted list containing a copy of the entries in the
        // given dictionary. The elements of the sorted list are ordered according
        // to the IComparable interface, which must be implemented by the
        // keys of all entries in the given dictionary as well as keys
        // subsequently added to the sorted list.
        // 
        public SortedList(IDictionary<TKey, TValue> dictionary)
            : this(dictionary, null)
        {
        }

        // Constructs a new sorted list containing a copy of the entries in the
        // given dictionary. The elements of the sorted list are ordered according
        // to the given IComparer implementation. If comparer is
        // null, the elements are compared to each other using the
        // IComparable interface, which in that case must be implemented
        // by the keys of all entries in the given dictionary as well as keys
        // subsequently added to the sorted list.
        // 
        public SortedList(IDictionary<TKey, TValue> dictionary, IComparer<TKey> comparer)
            : this(dictionary?.Count ?? 0, comparer)
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            var count = dictionary.Count;
            if (count != 0)
            {
                dictionary.Keys.CopyTo(keys, 0);
                dictionary.Values.CopyTo(values, 0);
                Debug.Assert(count == keys.Length);
                if (count > 1)
                {
                    comparer = Comparer; // obtain default if this is null.
                    Array.Sort(keys, values, comparer);
                    for (var i = 1; i != keys.Length; ++i)
                    {
                        if (comparer.Compare(keys[i - 1], keys[i]) == 0)
                        {
                            throw new ArgumentException();
                        }
                    }
                }
            }

            _size = count;
        }
        
        public SortedList(SortedList<TKey, TValue> dictionary)
            : this(dictionary?.Count ?? 0, dictionary?.Comparer)
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            var count = dictionary.Count;
            if (count != 0)
            {
                dictionary.Keys.CopyTo(keys, 0);
                dictionary.Values.CopyTo(values, 0);
                Debug.Assert(count == keys.Length);
            }

            _size = count;
        }

        // Adds an entry with the given key and value to this sorted list. An
        // ArgumentException is thrown if the key is already present in the sorted list.
        // 
        public void Add(TKey key, TValue value)
        {
            if (key == null) throw new ArgumentNullException(nameof(key));
            var i = Array.BinarySearch(keys, 0, _size, key, comparer);
            if (i >= 0)
                throw new ArgumentException(nameof(key));
            Insert(~i, key, value);
        }

        void ICollection<KeyValuePair<TKey, TValue>>.Add(KeyValuePair<TKey, TValue> keyValuePair)
        {
            Add(keyValuePair.Key, keyValuePair.Value);
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Contains(KeyValuePair<TKey, TValue> keyValuePair)
        {
            var index = IndexOfKey(keyValuePair.Key);
            if (index >= 0 && EqualityComparer<TValue>.Default.Equals(values[index], keyValuePair.Value))
            {
                return true;
            }

            return false;
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Remove(KeyValuePair<TKey, TValue> keyValuePair)
        {
            var index = IndexOfKey(keyValuePair.Key);
            if (index >= 0 && EqualityComparer<TValue>.Default.Equals(values[index], keyValuePair.Value))
            {
                RemoveAt(index);
                return true;
            }

            return false;
        }

        // Returns the capacity of this sorted list. The capacity of a sorted list
        // represents the allocated length of the internal arrays used to store the
        // keys and values of the list, and thus also indicates the maximum number
        // of entries the list can contain before a reallocation of the internal
        // arrays is required.
        // 
        public int Capacity
        {
            get => keys.Length;
            set
            {
                if (value != keys.Length)
                {
                    if (value < _size)
                    {
                        throw new ArgumentOutOfRangeException(nameof(value));
                    }

                    if (value > 0)
                    {
                        var newKeys = new TKey[value];
                        var newValues = new TValue[value];
                        if (_size > 0)
                        {
                            Array.Copy(keys, 0, newKeys, 0, _size);
                            Array.Copy(values, 0, newValues, 0, _size);
                        }

                        keys = newKeys;
                        values = newValues;
                    }
                    else
                    {
                        keys = Array.Empty<TKey>();
                        values = Array.Empty<TValue>();
                    }
                }
            }
        }

        public IComparer<TKey> Comparer => comparer;

        void IDictionary.Add(object key, object value)
        {
            if (key == null)
                throw new ArgumentNullException(nameof(key));

            if (value == null && !(default(TValue) == null)) // null is an invalid value for Value types
                throw new ArgumentNullException(nameof(value));

            if (!(key is TKey))
                throw new ArgumentException(nameof(key));

            if (!(value is TValue) && value != null) // null is a valid value for Reference Types
                throw new ArgumentException(nameof(value));

            Add((TKey) key, (TValue) value);
        }

        // Returns the number of entries in this sorted list.
        public int Count => _size;

        // Returns a collection representing the keys of this sorted list. This
        // method returns the same object as GetKeyList, but typed as an
        // ICollection instead of an IList.
        public IList<TKey> Keys => GetKeyListHelper();

        ICollection<TKey> IDictionary<TKey, TValue>.Keys => GetKeyListHelper();

        ICollection IDictionary.Keys => GetKeyListHelper();

        IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => GetKeyListHelper();

        // Returns a collection representing the values of this sorted list. This
        // method returns the same object as GetValueList, but typed as an
        // ICollection instead of an IList.
        // 
        public IList<TValue> Values => GetValueListHelper();

        ICollection<TValue> IDictionary<TKey, TValue>.Values => GetValueListHelper();

        ICollection IDictionary.Values => GetValueListHelper();

        IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => GetValueListHelper();

        private KeyList GetKeyListHelper()
        {
            if (keyList == null)
                keyList = new KeyList(this);
            return keyList;
        }

        private ValueList GetValueListHelper()
        {
            if (valueList == null)
                valueList = new ValueList(this);
            return valueList;
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.IsReadOnly => false;

        bool IDictionary.IsReadOnly => false;

        bool IDictionary.IsFixedSize => false;

        bool ICollection.IsSynchronized => false;

        // Synchronization root for this object.
        object ICollection.SyncRoot => this;

        // Removes all entries from this sorted list.
        public void Clear()
        {
            // clear does not change the capacity
            version++;
            _size = 0;
        }

        bool IDictionary.Contains(object key)
        {
            if (IsCompatibleKey(key))
            {
                return ContainsKey((TKey) key);
            }

            return false;
        }

        // Checks if this sorted list contains an entry with the given key.
        public bool ContainsKey(TKey key)
        {
            return IndexOfKey(key) >= 0;
        }

        // Checks if this sorted list contains an entry with the given value. The
        // values of the entries of the sorted list are compared to the given value
        // using the Object.Equals method. This method performs a linear
        // search and is substantially slower than the Contains
        // method.
        public bool ContainsValue(TValue value)
        {
            return IndexOfValue(value) >= 0;
        }

        // Copies the values in this SortedList to an array.
        void ICollection<KeyValuePair<TKey, TValue>>.CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            if (arrayIndex < 0 || arrayIndex > array.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(arrayIndex));
            }

            if (array.Length - arrayIndex < Count)
            {
                throw new ArgumentException();
            }

            for (var i = 0; i < Count; i++)
            {
                var entry = new KeyValuePair<TKey, TValue>(keys[i], values[i]);
                array[arrayIndex + i] = entry;
            }
        }

        void ICollection.CopyTo(Array array, int index)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            if (array.Rank != 1)
            {
                throw new ArgumentException(nameof(array));
            }

            if (array.GetLowerBound(0) != 0)
            {
                throw new ArgumentException(nameof(array));
            }

            if (index < 0 || index > array.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            if (array.Length - index < Count)
            {
                throw new ArgumentException();
            }

            var keyValuePairArray = array as KeyValuePair<TKey, TValue>[];
            if (keyValuePairArray != null)
            {
                for (var i = 0; i < Count; i++)
                {
                    keyValuePairArray[i + index] = new KeyValuePair<TKey, TValue>(keys[i], values[i]);
                }
            }
            else
            {
                var objects = array as object[];
                if (objects == null)
                {
                    throw new ArgumentException(nameof(array));
                }

                try
                {
                    for (var i = 0; i < Count; i++)
                    {
                        objects[i + index] = new KeyValuePair<TKey, TValue>(keys[i], values[i]);
                    }
                }
                catch (ArrayTypeMismatchException)
                {
                    throw new ArgumentException(nameof(array));
                }
            }
        }

        private const int MaxArrayLength = 0X7FEFFFFF;

        // Ensures that the capacity of this sorted list is at least the given
        // minimum value. If the current capacity of the list is less than
        // min, the capacity is increased to twice the current capacity or
        // to min, whichever is larger.
        private void EnsureCapacity(int min)
        {
            var newCapacity = keys.Length == 0 ? DefaultCapacity : keys.Length * 2;
            // Allow the list to grow to maximum possible capacity (~2G elements) before encountering overflow.
            // Note that this check works even when _items.Length overflowed thanks to the (uint) cast
            if ((uint) newCapacity > MaxArrayLength) newCapacity = MaxArrayLength;
            if (newCapacity < min) newCapacity = min;
            Capacity = newCapacity;
        }

        // Returns the value of the entry at the given index.
        private TValue GetByIndex(int index)
        {
            if (index < 0 || index >= _size)
                throw new ArgumentOutOfRangeException(nameof(index));
            return values[index];
        }

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            return new Enumerator(this, Enumerator.KeyValuePair);
        }

        IEnumerator<KeyValuePair<TKey, TValue>> IEnumerable<KeyValuePair<TKey, TValue>>.GetEnumerator()
        {
            return new Enumerator(this, Enumerator.KeyValuePair);
        }

        IDictionaryEnumerator IDictionary.GetEnumerator()
        {
            return new Enumerator(this, Enumerator.DictEntry);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return new Enumerator(this, Enumerator.KeyValuePair);
        }

        // Returns the key of the entry at the given index.
        private TKey GetKey(int index)
        {
            if (index < 0 || index >= _size)
                throw new ArgumentOutOfRangeException(nameof(index));
            return keys[index];
        }

        // Returns the value associated with the given key. If an entry with the
        // given key is not found, the returned value is null.
        public TValue this[TKey key]
        {
            get
            {
                var i = IndexOfKey(key);
                if (i >= 0)
                    return values[i];

                throw new KeyNotFoundException(key.ToString());
            }
            set
            {
                if (key == null) throw new ArgumentNullException(nameof(key));
                var i = Array.BinarySearch(keys, 0, _size, key, comparer);
                if (i >= 0)
                {
                    values[i] = value;
                    version++;
                    return;
                }

                Insert(~i, key, value);
            }
        }

        object IDictionary.this[object key]
        {
            get
            {
                if (IsCompatibleKey(key))
                {
                    var i = IndexOfKey((TKey) key);
                    if (i >= 0)
                    {
                        return values[i];
                    }
                }

                return null;
            }
            set
            {
                if (!IsCompatibleKey(key))
                {
                    throw new ArgumentNullException(nameof(key));
                }

                if (value == null && !(default(TValue) == null))
                    throw new ArgumentNullException(nameof(value));

                var tempKey = (TKey) key;
                try
                {
                    this[tempKey] = (TValue) value;
                }
                catch (InvalidCastException)
                {
                    throw new ArgumentException(nameof(value));
                }
            }
        }

        // Returns the index of the entry with a given key in this sorted list. The
        // key is located through a binary search, and thus the average execution
        // time of this method is proportional to Log2(size), where
        // size is the size of this sorted list. The returned value is -1 if
        // the given key does not occur in this sorted list. Null is an invalid 
        // key value.
        public int IndexOfKey(TKey key)
        {
            if (key == null)
                throw new ArgumentNullException(nameof(key));
            var ret = Array.BinarySearch(keys, 0, _size, key, comparer);
            return ret >= 0 ? ret : -1;
        }

        // Returns the index of the first occurrence of an entry with a given value
        // in this sorted list. The entry is located through a linear search, and
        // thus the average execution time of this method is proportional to the
        // size of this sorted list. The elements of the list are compared to the
        // given value using the Object.Equals method.
        public int IndexOfValue(TValue value)
        {
            return Array.IndexOf(values, value, 0, _size);
        }

        // Inserts an entry with a given key and value at a given index.
        private void Insert(int index, TKey key, TValue value)
        {
            if (_size == keys.Length) EnsureCapacity(_size + 1);
            if (index < _size)
            {
                Array.Copy(keys, index, keys, index + 1, _size - index);
                Array.Copy(values, index, values, index + 1, _size - index);
            }

            keys[index] = key;
            values[index] = value;
            _size++;
            version++;
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            var i = IndexOfKey(key);
            if (i >= 0)
            {
                value = values[i];
                return true;
            }

            value = default;
            return false;
        }

        // Removes the entry at the given index. The size of the sorted list is
        // decreased by one.
        public void RemoveAt(int index)
        {
            if (index < 0 || index >= _size)
                throw new ArgumentOutOfRangeException(nameof(index));
            _size--;
            if (index < _size)
            {
                Array.Copy(keys, index + 1, keys, index, _size - index);
                Array.Copy(values, index + 1, values, index, _size - index);
            }

            version++;
        }

        // Removes an entry from this sorted list. If an entry with the specified
        // key exists in the sorted list, it is removed. An ArgumentException is
        // thrown if the key is null.
        public bool Remove(TKey key)
        {
            var i = IndexOfKey(key);
            if (i >= 0)
                RemoveAt(i);
            return i >= 0;
        }

        void IDictionary.Remove(object key)
        {
            if (IsCompatibleKey(key))
            {
                Remove((TKey) key);
            }
        }

        // Sets the capacity of this sorted list to the size of the sorted list.
        // This method can be used to minimize a sorted list's memory overhead once
        // it is known that no new elements will be added to the sorted list. To
        // completely clear a sorted list and release all memory referenced by the
        // sorted list, execute the following statements:
        // 
        // SortedList.Clear();
        // SortedList.TrimExcess();
        public void TrimExcess()
        {
            var threshold = (int) (keys.Length * 0.9);
            if (_size < threshold)
            {
                Capacity = _size;
            }
        }

        private static bool IsCompatibleKey(object key)
        {
            if (key == null)
            {
                throw new ArgumentNullException(nameof(key));
            }

            return (key is TKey);
        }

        private struct Enumerator : IEnumerator<KeyValuePair<TKey, TValue>>, IDictionaryEnumerator
        {
            private SortedList<TKey, TValue> _sortedList;
            private TKey _key;
            private TValue _value;
            private int _index;
            private int _version;
            private int _getEnumeratorRetType; // What should Enumerator.Current return?

            internal const int KeyValuePair = 1;
            internal const int DictEntry = 2;

            internal Enumerator(SortedList<TKey, TValue> sortedList, int getEnumeratorRetType)
            {
                _sortedList = sortedList;
                _index = 0;
                _version = _sortedList.version;
                _getEnumeratorRetType = getEnumeratorRetType;
                _key = default;
                _value = default;
            }

            public void Dispose()
            {
                _index = 0;
                _key = default;
                _value = default;
            }

            object IDictionaryEnumerator.Key
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    return _key;
                }
            }

            public bool MoveNext()
            {
                if (_version != _sortedList.version)
                    throw new InvalidOperationException();

                if ((uint) _index < (uint) _sortedList.Count)
                {
                    _key = _sortedList.keys[_index];
                    _value = _sortedList.values[_index];
                    _index++;
                    return true;
                }

                _index = _sortedList.Count + 1;
                _key = default;
                _value = default;
                return false;
            }

            DictionaryEntry IDictionaryEnumerator.Entry
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    return new DictionaryEntry(_key, _value);
                }
            }

            public KeyValuePair<TKey, TValue> Current => new KeyValuePair<TKey, TValue>(_key, _value);

            object IEnumerator.Current
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    if (_getEnumeratorRetType == DictEntry)
                    {
                        return new DictionaryEntry(_key, _value);
                    }
                    else
                    {
                        return new KeyValuePair<TKey, TValue>(_key, _value);
                    }
                }
            }

            object IDictionaryEnumerator.Value
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    return _value;
                }
            }

            void IEnumerator.Reset()
            {
                if (_version != _sortedList.version)
                {
                    throw new InvalidOperationException();
                }

                _index = 0;
                _key = default;
                _value = default;
            }
        }

        private sealed class SortedListKeyEnumerator : IEnumerator<TKey>
        {
            private SortedList<TKey, TValue> _sortedList;
            private int _index;
            private int _version;
            private TKey _currentKey;

            internal SortedListKeyEnumerator(SortedList<TKey, TValue> sortedList)
            {
                _sortedList = sortedList;
                _version = sortedList.version;
            }

            public void Dispose()
            {
                _index = 0;
                _currentKey = default;
            }

            public bool MoveNext()
            {
                if (_version != _sortedList.version)
                {
                    throw new InvalidOperationException();
                }

                if ((uint) _index < (uint) _sortedList.Count)
                {
                    _currentKey = _sortedList.keys[_index];
                    _index++;
                    return true;
                }

                _index = _sortedList.Count + 1;
                _currentKey = default;
                return false;
            }

            public TKey Current => _currentKey;

            object IEnumerator.Current
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    return _currentKey;
                }
            }

            void IEnumerator.Reset()
            {
                if (_version != _sortedList.version)
                {
                    throw new InvalidOperationException();
                }

                _index = 0;
                _currentKey = default;
            }
        }

        private sealed class SortedListValueEnumerator : IEnumerator<TValue>
        {
            private SortedList<TKey, TValue> _sortedList;
            private int _index;
            private int _version;
            private TValue _currentValue;

            internal SortedListValueEnumerator(SortedList<TKey, TValue> sortedList)
            {
                _sortedList = sortedList;
                _version = sortedList.version;
            }

            public void Dispose()
            {
                _index = 0;
                _currentValue = default;
            }

            public bool MoveNext()
            {
                if (_version != _sortedList.version)
                {
                    throw new InvalidOperationException();
                }

                if ((uint) _index < (uint) _sortedList.Count)
                {
                    _currentValue = _sortedList.values[_index];
                    _index++;
                    return true;
                }

                _index = _sortedList.Count + 1;
                _currentValue = default;
                return false;
            }

            public TValue Current => _currentValue;

            object IEnumerator.Current
            {
                get
                {
                    if (_index == 0 || (_index == _sortedList.Count + 1))
                    {
                        throw new InvalidOperationException();
                    }

                    return _currentValue;
                }
            }

            void IEnumerator.Reset()
            {
                if (_version != _sortedList.version)
                {
                    throw new InvalidOperationException();
                }

                _index = 0;
                _currentValue = default;
            }
        }

        [DebuggerTypeProxy(typeof(DictionaryKeyCollectionDebugView<>))]
        [DebuggerDisplay("Count = {Count}")]
        [Serializable]
        public sealed class KeyList : IList<TKey>, ICollection
        {
            private SortedList<TKey, TValue> _dict; // Do not rename (binary serialization)

            internal KeyList(SortedList<TKey, TValue> dictionary)
            {
                _dict = dictionary;
            }

            public int Count => _dict._size;

            public bool IsReadOnly => true;

            bool ICollection.IsSynchronized => false;

            object ICollection.SyncRoot => ((ICollection) _dict).SyncRoot;

            public void Add(TKey key)
            {
                throw new NotSupportedException();
            }

            public void Clear()
            {
                throw new NotSupportedException();
            }

            public bool Contains(TKey key)
            {
                return _dict.ContainsKey(key);
            }

            public void CopyTo(TKey[] array, int arrayIndex)
            {
                // defer error checking to Array.Copy
                Array.Copy(_dict.keys, 0, array, arrayIndex, _dict.Count);
            }

            void ICollection.CopyTo(Array array, int arrayIndex)
            {
                if (array != null && array.Rank != 1)
                    throw new ArgumentException(nameof(array));

                try
                {
                    // defer error checking to Array.Copy
                    Array.Copy(_dict.keys, 0, array, arrayIndex, _dict.Count);
                }
                catch (ArrayTypeMismatchException)
                {
                    throw new ArgumentException(nameof(array));
                }
            }

            public void Insert(int index, TKey value)
            {
                throw new NotSupportedException();
            }

            public TKey this[int index]
            {
                get => _dict.GetKey(index);
                set => throw new NotSupportedException();
            }

            public IEnumerator<TKey> GetEnumerator()
            {
                return new SortedListKeyEnumerator(_dict);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return new SortedListKeyEnumerator(_dict);
            }

            public int IndexOf(TKey key)
            {
                if (key == null)
                    throw new ArgumentNullException(nameof(key));

                var i = Array.BinarySearch(_dict.keys, 0,
                    _dict.Count, key, _dict.comparer);
                if (i >= 0) return i;
                return -1;
            }

            public bool Remove(TKey key)
            {
                throw new NotSupportedException();
                // return false;
            }

            public void RemoveAt(int index)
            {
                throw new NotSupportedException();
            }
        }

        [DebuggerTypeProxy(typeof(DictionaryValueCollectionDebugView<>))]
        [DebuggerDisplay("Count = {Count}")]
        [Serializable]
        public sealed class ValueList : IList<TValue>, ICollection
        {
            private SortedList<TKey, TValue> _dict; // Do not rename (binary serialization)

            internal ValueList(SortedList<TKey, TValue> dictionary)
            {
                _dict = dictionary;
            }

            public int Count => _dict._size;

            public bool IsReadOnly => true;

            bool ICollection.IsSynchronized => false;

            object ICollection.SyncRoot => ((ICollection) _dict).SyncRoot;

            public void Add(TValue key)
            {
                throw new NotSupportedException();
            }

            public void Clear()
            {
                throw new NotSupportedException();
            }

            public bool Contains(TValue value)
            {
                return _dict.ContainsValue(value);
            }

            public void CopyTo(TValue[] array, int arrayIndex)
            {
                // defer error checking to Array.Copy
                Array.Copy(_dict.values, 0, array, arrayIndex, _dict.Count);
            }

            void ICollection.CopyTo(Array array, int index)
            {
                if (array != null && array.Rank != 1)
                    throw new ArgumentException(nameof(array));

                try
                {
                    // defer error checking to Array.Copy
                    Array.Copy(_dict.values, 0, array, index, _dict.Count);
                }
                catch (ArrayTypeMismatchException)
                {
                    throw new ArgumentException(nameof(array));
                }
            }

            public void Insert(int index, TValue value)
            {
                throw new NotSupportedException();
            }

            public TValue this[int index]
            {
                get => _dict.GetByIndex(index);
                set => throw new NotSupportedException();
            }

            public IEnumerator<TValue> GetEnumerator()
            {
                return new SortedListValueEnumerator(_dict);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return new SortedListValueEnumerator(_dict);
            }

            public int IndexOf(TValue value)
            {
                return Array.IndexOf(_dict.values, value, 0, _dict.Count);
            }

            public bool Remove(TValue value)
            {
                throw new NotSupportedException();
                // return false;
            }

            public void RemoveAt(int index)
            {
                throw new NotSupportedException();
            }
        }
    }
}