using System.Collections.Generic;

namespace Tests
{
    public static class CollectionsExtensions
    {
        public static void ChangeCollection<T>(this ICollection<T> collection, IEnumerable<T> removedItems, IEnumerable<T> newItems)
        {
            foreach (var removedItem in removedItems)
            {
                collection.Remove(removedItem);
            }

            foreach (var newItem in newItems)
            {
                collection.Add(newItem);
            }
        }
    }
}