using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using RDotNet;

namespace StatsLab
{
    public class Experiment : IDisposable
    {
        private readonly REngine _rEngine;
        private readonly Dictionary<string, List<float>> _data = new Dictionary<string, List<float>>();
        
        public Experiment(List<string> variableNames)
        {
            if (variableNames == null) throw new ArgumentNullException(nameof(variableNames));
            
            try
            {
                _rEngine = REngine.GetInstance();
            }
            catch (Exception e)
            {
                throw new ApplicationException(
                    "Failed to create R engine. Make sure R (https://www.r-project.org) is properly installed.", e);
            }

            foreach (var name in variableNames)
            {
                _data.Add(name, new List<float>());
            }
        }

        public void AddValue(string name, float value)
        {
            if (!_data.ContainsKey(name))
            {
                throw new ArgumentOutOfRangeException(nameof(name));
            }
            _data[name].Add(value);
        }

        public void Finish()
        {
            foreach (var name in _data.Keys)
            {
                var vector = _rEngine.CreateNumericVector(_data[name].ConvertAll(x => (double)x));
                _rEngine.SetSymbol(name, vector);
                var swTestRes = _rEngine.Evaluate($"shapiro.test({name})").AsList();
                var pSw = swTestRes["p.value"].AsNumeric().First();
                if (pSw <= 0.05)
                {
                    Console.WriteLine($"WARNING: According to Shapiro-Wilk Normality Test the samples in the '{name}' group didn't come from a Normal distribution.");
                }
                
                var assemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                var fileName = Path.Combine(assemblyDir, name).Replace('\\','/');
                if (File.Exists(fileName))
                {
                    _ = _rEngine.Evaluate($"df<-read.table(\"{fileName}\", header=FALSE)");
                    var tTestRes = _rEngine.Evaluate($"t.test(df, {name})");
                    var pT = tTestRes.AsList()["p.value"].AsNumeric().First();
                    var prevMean = _rEngine.Evaluate("colMeans(df)").AsList()["V1"].AsNumeric().First();
                    var currMean = _rEngine.Evaluate($"mean({name})").AsNumeric().First();
                    if (pT > 0.05)
                    {
                        Console.WriteLine($"There is no statistically significant difference in current and previous means of '{name}' variables.");
                    }
                    else
                    {
                        Console.WriteLine($"There is a statistically significant difference in current and previous means of '{name}' variables.");
                        Console.WriteLine($"Previous mean: {prevMean}, current mean: {currMean}");
                    }

                    File.Delete(fileName);
                }

                File.WriteAllLines(fileName,
                    _data[name].ConvertAll(x => x.ToString(CultureInfo.InvariantCulture)));
            }
            Dispose();
        }

        public void Dispose()
        {
            _rEngine.Dispose();
        }
    }
}