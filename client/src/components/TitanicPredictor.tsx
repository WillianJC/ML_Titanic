import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import {
  UserIcon,
  CurrencyDollarIcon,
  CalendarIcon,
  UsersIcon,
  UserGroupIcon,
  TicketIcon,
  BoltIcon,
  ArrowPathIcon,
  MinusIcon,
  PlusIcon,
} from "@heroicons/react/24/outline";
import "../styles/titanic.css";

const MODEL_URL = "/model/titanic_model_js/model.json";

type Inputs = {
  pclass: number;
  sex: number;
  age: number;
  sibsp: number;
  parch: number;
  fare: number;
};

export default function TitanicPredictor(): React.JSX.Element {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Inputs>({
    pclass: 3,
    sex: 0,
    age: 30,
    sibsp: 0,
    parch: 0,
    fare: 7.25,
  });
  const [result, setResult] = useState<number | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);

    // Carga el modelo de TF.js desde la URL pública.
    tf.loadLayersModel(MODEL_URL)
      .then((m) => {
        if (!mounted) return;
        setModel(m);
        setError(null);
      })
      .catch((err) => setError(String(err)))
      .finally(() => mounted && setLoading(false));

    return () => {
      mounted = false;
    };
  }, []);

  // Maneja cambios en los inputs del formulario.
  const handleChange =
    (field: keyof Inputs) =>
    (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      const value = e.target.value;
      setInputs((s) => ({
        ...s,
        [field]:
          field === "sex" || field === "pclass"
            ? parseInt(value)
            : parseFloat(value),
      }));
    };

  const handleIncrement = (field: keyof Inputs, amount: number) => () => {
    setInputs((s) => {
      const newValue = s[field] + amount;
      if (newValue < 0) return s;
      return { ...s, [field]: newValue };
    });
  };

  // Realiza la predicción usando el modelo cargado y los inputs del usuario.
  const predict = async () => {
    if (!model) return;
    const arr = [
      inputs.pclass,
      inputs.sex,
      inputs.age || 0,
      inputs.sibsp || 0,
      inputs.parch || 0,
      inputs.fare || 0,
    ];

    // Min/max usados en el entrenamiento.
    const TRAIN_MIN = [1.0, 0, 0.42, 0.0, 0.0, 0.0];
    const TRAIN_MAX = [3.0, 1, 80.0, 8.0, 6.0, 512.3292];

    // Aplica (v - min) / (max - min) por columna.
    const arrNormalized = arr.map((v, i) => {
      const min = TRAIN_MIN[i];
      const max = TRAIN_MAX[i];

      if (max === min) return 0;
      return (v - min) / (max - min);
    });

    const x = tf.tensor2d([arrNormalized], [1, 6], "float32");
    try {
      const y = (model.predict(x) as tf.Tensor) || null; // Realiza la predicción con el modelo.
      if (y) {
        const data = await (y as tf.Tensor).data(); // Obtiene los datos del tensor de salida.
        const prob = Array.from(data)[0]; // Probabilidad de supervivencia.
        setResult(prob);
      }
    } catch (e) {
      setError(String(e)); // Maneja errores de predicción.
    } finally {
      x.dispose(); // Limpia el tensor para liberar memoria.
    }
  };

  return (
    <section className="titanic">
      <h2>Titanic Predictor (TensorFlow.js)</h2>

      {loading ? (
        <div className="status">Cargando modelo…</div>
      ) : error ? (
        <div className="status error">Error cargando modelo: {error}</div>
      ) : (
        <div className="status success">
          Modelo cargado — listo para predecir
        </div>
      )}

      <div className="form">
        <label>
          <div className="label-text">
            <TicketIcon className="icon" />
            Clase de Pasajero
          </div>
          <select value={inputs.pclass} onChange={handleChange("pclass")}>
            <option value={1}>Primera Clase</option>
            <option value={2}>Segunda Clase</option>
            <option value={3}>Tercera Clase</option>
          </select>
        </label>

        <label>
          <div className="label-text">
            <UserIcon className="icon" />
            Genero
          </div>
          <select value={inputs.sex} onChange={handleChange("sex")}>
            <option value={0}>Masculino</option>
            <option value={1}>Femenino</option>
          </select>
        </label>

        <label>
          <div className="label-text">
            <CalendarIcon className="icon" />
            Edad
          </div>
          <div className="number-control">
            <button onClick={handleIncrement("age", -1)}>
              <MinusIcon className="icon-small" />
            </button>
            <input
              type="number"
              step="1"
              min="0"
              value={inputs.age}
              onChange={handleChange("age")}
            />
            <button onClick={handleIncrement("age", 1)}>
              <PlusIcon className="icon-small" />
            </button>
          </div>
        </label>

        <label>
          <div className="label-text">
            <UsersIcon className="icon" />
            Hermanes
          </div>
          <div className="number-control">
            <button onClick={handleIncrement("sibsp", -1)}>
              <MinusIcon className="icon-small" />
            </button>
            <input
              type="number"
              min="0"
              step="1"
              value={inputs.sibsp}
              onChange={handleChange("sibsp")}
            />
            <button onClick={handleIncrement("sibsp", 1)}>
              <PlusIcon className="icon-small" />
            </button>
          </div>
        </label>

        <label>
          <div className="label-text">
            <UserGroupIcon className="icon" />
            Padres
          </div>
          <div className="number-control">
            <button onClick={handleIncrement("parch", -1)}>
              <MinusIcon className="icon-small" />
            </button>
            <input
              type="number"
              min="0"
              step="1"
              value={inputs.parch}
              onChange={handleChange("parch")}
            />
            <button onClick={handleIncrement("parch", 1)}>
              <PlusIcon className="icon-small" />
            </button>
          </div>
        </label>

        <label>
          <div className="label-text">
            <CurrencyDollarIcon className="icon" />
            Tarifa
          </div>
          <input
            type="number"
            step="0.01"
            value={inputs.fare}
            onChange={handleChange("fare")}
          />
        </label>

        <div className="actions">
          <button onClick={predict} disabled={!model || loading}>
            <BoltIcon className="icon" />
            Predecir
          </button>
          <button
            onClick={() => {
              setInputs({
                pclass: 3,
                sex: 0,
                age: 30,
                sibsp: 0,
                parch: 0,
                fare: 7.25,
              });
              setResult(null);
            }}
          >
            <ArrowPathIcon className="icon" />
            Reset
          </button>
        </div>

        <div className="result-container">
          {result !== null ? (
            <div className="result">
              <strong>Probabilidad de supervivencia:</strong>{" "}
              {(result * 100).toFixed(2)}% —{" "}
              <em>{result >= 0.5 ? "Sobrevive" : "No sobrevive"}</em>
            </div>
          ) : (
            <div className="result placeholder">
              Esperando datos para predecir...
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
