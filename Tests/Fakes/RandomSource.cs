using System;
using System.Collections.Generic;
using System.Reflection;
using Redzen.Random;
using Xunit;

namespace Tests.Fakes
{
    public class RandomSource : IRandomSource, IDisposable
    {
        private RandomSource(IRandomSource trueSource)
        {
            Assert.NotNull(trueSource);
            _innerSource = trueSource;
        }
            
        #region IRandomSource
        
        public void Reinitialise(ulong seed)
        {
        }
        
        public float NextFloat()
        {
            if (!_isParanoid)
                return _innerSource.NextFloat();
            
            Assert.True(_nextFloatsQueue.TryDequeue(out var value), "NextFloat call when queue is empty");
            return value;
        }

        public int Next()
        {
            if (!_isParanoid)
                return _innerSource.Next();
            
            Assert.True(_nextsQueue.TryDequeue(out var value), "Next call when queue is empty");
            return value;
        }

        public int Next(int maxValue)
        {
            if (!_isParanoid)
                return _innerSource.Next(maxValue);
            
            Assert.True(_nextsLimitedQueue.TryDequeue(out var value), "Next(max) call when queue is empty");
            Assert.True(value < maxValue, "Next(max) exceeds maxValue");
            return value;
        }
        
        public bool NextBool()
        {
            if (!_isParanoid)
                return _innerSource.NextBool();
            
            Assert.True(_nextBoolsQueue.TryDequeue(out var value), "NextBool call when queue is empty");
            return value;
        }

        public int Next(int minValue, int maxValue)
        {
            throw new NotImplementedException();
        }

        public double NextDouble()
        {
            throw new NotImplementedException();
        }

        public double NextDoubleHighRes()
        {
            throw new NotImplementedException();
        }

        public void NextBytes(byte[] buffer)
        {
            throw new NotImplementedException();
        }

        public uint NextUInt()
        {
            throw new NotImplementedException();
        }

        public int NextInt()
        {
            throw new NotImplementedException();
        }

        public ulong NextULong()
        {
            throw new NotImplementedException();
        }

        public double NextDoubleNonZero()
        {
            throw new NotImplementedException();
        }

        public byte NextByte()
        {
            throw new NotImplementedException();
        }
        #endregion

        internal static RandomSource DrillIn()
        {
            var currentSource = Neat.Utils.RandomSource.Rng;
            
            var field = typeof(Neat.Utils.RandomSource).GetField(nameof(Neat.Utils.RandomSource.Rng), BindingFlags.Static | BindingFlags.Public);
            var source = new RandomSource(currentSource);
            field.SetValue(null, source);
            return source;
        }

        internal void PushNextFloats(params float[] values)
        {
            foreach (var value in values)
            {
                _nextFloatsQueue.Enqueue(value);
            }
        }
        
        internal void PushNexts(params int[] values)
        {
            foreach (var value in values)
            {
                _nextsQueue.Enqueue(value);
            }
        }
        
        internal void PushLimitedNexts(params int[] values)
        {
            foreach (var value in values)
            {
                _nextsLimitedQueue.Enqueue(value);
            }
        }
        
        internal void PushNextBools(params bool[] values)
        {
            foreach (var value in values)
            {
                _nextBoolsQueue.Enqueue(value);
            }
        }

        public void GetParanoid()
        {
            _isParanoid = true;
        }
        
        public void Dispose()
        {
            var field = typeof(Neat.Utils.RandomSource).GetField(nameof(Neat.Utils.RandomSource.Rng), BindingFlags.Static | BindingFlags.Public);
            field.SetValue(null, _innerSource);
        }

        private bool _isParanoid;
        private readonly IRandomSource _innerSource;
        private readonly Queue<float> _nextFloatsQueue = new Queue<float>();
        private readonly Queue<int> _nextsQueue = new Queue<int>();
        private readonly Queue<int> _nextsLimitedQueue = new Queue<int>();
        private readonly Queue<bool> _nextBoolsQueue = new Queue<bool>();
    }
}