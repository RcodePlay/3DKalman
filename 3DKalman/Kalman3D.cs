using MathNet.Numerics.LinearAlgebra;

namespace _3DKalman;

public class Kalman3D
{
    private static readonly MatrixBuilder<double> M = Matrix<double>.Build;
    private static readonly VectorBuilder<double> V = Vector<double>.Build;

    private readonly Matrix<double> _measurementModel;     // Extracts sensor data: px, py, pz
    private readonly Matrix<double> _sensorNoise;
    private readonly Matrix<double> _modelNoise = M.DenseIdentity(6);    // Difference in velocity change
    private readonly double _accelVar;

    private Vector<double> X { get; set; }   // [px,py,pz,vx,vy,vz]
    private Matrix<double> FilterUncertainty { get; set; }

    public Kalman3D(double measStd, double accelStd, Vector<double> initialPos)
    {
        _accelVar = accelStd * accelStd;

        _measurementModel = M.Dense(3, 6);
        for (var i = 0; i < 3; i++) _measurementModel[i, i] = 1;

        _sensorNoise = M.DenseIdentity(3) * (measStd * measStd);

        X = V.Dense([initialPos[0], initialPos[1], initialPos[2], 0, 0, 0]);

        // Position known to about the sensor noise, velocity unknown
        FilterUncertainty = M.DenseIdentity(6) * 1000.0;
        for (var i = 0; i < 3; i++) FilterUncertainty[i, i] = measStd * measStd;
    }

    private static Matrix<double> BuildMotionModel(double dt)
    {
        var motionModel = M.DenseIdentity(6);
        for (var i = 0; i < 3; i++) motionModel[i, i + 3] = dt;
        return motionModel;
    }

    private Matrix<double> BuildModelNoise(double dt)
    {
        var modelNoise = M.Dense(6, 6);
        double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
        for (var i = 0; i < 3; i++)
        {
            modelNoise[i, i]         = dt4 / 4 * _accelVar;
            modelNoise[i, i + 3]     = dt3 / 2 * _accelVar;
            modelNoise[i + 3, i]     = dt3 / 2 * _accelVar;
            modelNoise[i + 3, i + 3] = dt2 * _accelVar;
        }
        return modelNoise;
    }

    /// <summary>Advance the state by dt. Optional known acceleration (e.g., gravity).</summary>
    public void Predict(double dt, Vector<double>? knownAccel = null)
    {
        var motionModel = BuildMotionModel(dt);
        X = motionModel * X;

        if (knownAccel != null)
        {
            for (var i = 0; i < 3; i++)
            {
                X[i]     += 0.5 * knownAccel[i] * dt * dt;
                X[i + 3] += knownAccel[i] * dt;
            }
        }

        FilterUncertainty = motionModel * FilterUncertainty * motionModel.Transpose() + BuildModelNoise(dt);
    }

    /// <summary>Correct with a measured 3D position.</summary>
    public void Update(Vector<double> z)
    {
        var innovation = z - _measurementModel * X;
        var innovationCovariance = _measurementModel * FilterUncertainty * _measurementModel.Transpose() + _sensorNoise;
        var kalmanGain = FilterUncertainty * _measurementModel.Transpose() * innovationCovariance.Inverse();

        X += kalmanGain * innovation;

        // Joseph form: numerically safer than (I - KH)P
        var ikh = _modelNoise - kalmanGain * _measurementModel;
        FilterUncertainty = ikh * FilterUncertainty * ikh.Transpose() + kalmanGain * _sensorNoise * kalmanGain.Transpose();
    }

    /// <summary>Predict future positions without touching the filter state.</summary>
    public Vector<double>[] Forecast(int steps, double dt, Vector<double>? knownAccel = null)
    {
        var motionModel = BuildMotionModel(dt);
        var x = X.Clone();
        var result = new Vector<double>[steps];

        for (var s = 0; s < steps; s++)
        {
            x = motionModel * x;
            if (knownAccel != null)
            {
                for (var i = 0; i < 3; i++)
                {
                    x[i]     += 0.5 * knownAccel[i] * dt * dt;
                    x[i + 3] += knownAccel[i] * dt;
                }
            }
            result[s] = V.Dense([x[0], x[1], x[2]]);
        }
        return result;
    }
}