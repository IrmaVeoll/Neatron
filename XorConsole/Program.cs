using System;
using System.Collections.Generic;
using Neat;
using StatsLab;
using XorDemo.Model;

namespace XorConsole
{
    class Program
    {
        private static readonly PopulationParameters PopulationParameters =
            new PopulationParameters(300, 2);

        private static readonly NetworkParameters NetworkParameters =
            new NetworkParameters(2, 1, NetworkType.FeedForward)
            {
                InitialConnectionDensity = 1f
            };
        
        private static readonly ReproductionParameters ReproductionParameters =
            new ReproductionParameters
            {
                CrossoverType = CrossoverType.OnePoint,

                WeightMutations = new WeightMutations
                {
                    OverallRouletteWheelShare = 80,
                    Mutations =
                    {
                        new WeightTweak
                        {
                            RouletteWheelShare = 200f,
                            Sigma = 0.5f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 20f,
                            ConnectionCount = 2,
                            Sigma = 0.01f
                        },
                        new WeightTweak
                        {
                            RouletteWheelShare = 1f,
                            ConnectionCount = 3,
                            Sigma = 0.008f
                        },
                        new WeightPerturb
                        {
                            RouletteWheelShare = 2f
                        }
                    }
                },
                AddConnectionRouletteWheelShare = 10f,
                RemoveConnectionRouletteWheelShare = 10f,
                SplitConnectionRouletteWheelShare = 2f
            };

        static void Main(string[] args)
        {
            var exp = new Experiment(new List<string>{"fitness", "solGenNum"});
            
            for (var i = 0; i < 30; i++)
            {
                var searchModel = new XorNetworkSearch(PopulationParameters, NetworkParameters, ReproductionParameters);
                int solutionGenNum = 1001;
                for (var g = 0; g < 1000; g++)
                {
                    var res = searchModel.SearchNext();
                    if (res.FitnessRating[0].Fitness > 0.99 && g < solutionGenNum)
                    {
                        solutionGenNum = g;
                    }
                }

                var finRes = searchModel.SearchNext();
                exp.AddValue("solGenNum", solutionGenNum);
                exp.AddValue("fitness", finRes.FitnessRating[0].Fitness);
                
                Console.WriteLine($"{finRes.FitnessRating[0].Fitness}\t{solutionGenNum}");
            }
            exp.Finish();

            Console.WriteLine("Meow.");
        }
    }
}