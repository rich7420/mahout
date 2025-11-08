# QuMat Class Methods

This document provides comprehensive documentation for all methods in the QuMat class, following the reStructuredText format as specified in PEP 287.

## Table of Contents

- [Initialization](#initialization)
  - [`__init__(self, backend_config)`](#__initself-backend_config)
- [Circuit Creation](#circuit-creation)
  - [`create_empty_circuit(self, num_qubits)`](#create_empty_circuitself-num_qubits)
- [Basic Quantum Gates](#basic-quantum-gates)
  - [`apply_not_gate(self, qubit_index)`](#apply_not_gateself-qubit_index)
  - [`apply_hadamard_gate(self, qubit_index)`](#apply_hadamard_gateself-qubit_index)
  - [`apply_cnot_gate(self, control_qubit_index, target_qubit_index)`](#apply_cnot_gateself-control_qubit_index-target_qubit_index)
  - [`apply_toffoli_gate(self, control_qubit_index1, control_qubit_index2, target_qubit_index)`](#apply_toffoli_gateself-control_qubit_index1-control_qubit_index2-target_qubit_index)
  - [`apply_swap_gate(self, qubit_index1, qubit_index2)`](#apply_swap_gateself-qubit_index1-qubit_index2)
  - [`apply_cswap_gate(self, control_qubit_index, target_qubit_index1, target_qubit_index2)`](#apply_cswap_gateself-control_qubit_index-target_qubit_index1-target_qubit_index2)
- [Pauli Gates](#pauli-gates)
  - [`apply_pauli_x_gate(self, qubit_index)`](#apply_pauli_x_gateself-qubit_index)
  - [`apply_pauli_y_gate(self, qubit_index)`](#apply_pauli_y_gateself-qubit_index)
  - [`apply_pauli_z_gate(self, qubit_index)`](#apply_pauli_z_gateself-qubit_index)
- [Rotation Gates](#rotation-gates)
  - [`apply_rx_gate(self, qubit_index, angle)`](#apply_rx_gateself-qubit_index-angle)
  - [`apply_ry_gate(self, qubit_index, angle)`](#apply_ry_gateself-qubit_index-angle)
  - [`apply_rz_gate(self, qubit_index, angle)`](#apply_rz_gateself-qubit_index-angle)
- [Universal Gate](#universal-gate)
  - [`apply_u_gate(self, qubit_index, theta, phi, lambd)`](#apply_u_gateself-qubit_index-theta-phi-lambd)
- [Circuit Execution](#circuit-execution)
  - [`execute_circuit(self, parameter_values=None)`](#execute_circuitself-parameter_valuesnone)
- [Parameter Binding](#parameter-binding)
  - [`bind_parameters(self, parameter_values)`](#bind_parametersself-parameter_values)
- [Circuit Visualization](#circuit-visualization)
  - [`draw(self)`](#drawself)
- [Quantum State Measurement](#quantum-state-measurement)
  - [`swap_test(self, ancilla_qubit, qubit1, qubit2)`](#swap_testself-ancilla_qubit-qubit1-qubit2)
  - [`measure_overlap(self, qubit1, qubit2, ancilla_qubit=0)`](#measure_overlapself-qubit1-qubit2-ancilla_qubit0)
  - [`calculate_prob_zero(self, results, ancilla_qubit)`](#calculate_prob_zeroself-results-ancilla_qubit)
- [Testing and Debugging](#testing-and-debugging)
  - [`get_final_state_vector(self)`](#get_final_state_vectorself)

## Initialization

### `__init__(self, backend_config)`

Initialize the QuMat instance with a specific quantum computing backend.

**Parameters:**

- **backend_config** (dict): Configuration dictionary for the backend. Must contain:
  
  - ``backend_name`` (str): Name of the backend to use (e.g., ``"qiskit"``, ``"cirq"``, ``"braket"``)
  
  - ``backend_options`` (dict): Backend-specific options such as:
    
    - ``simulator_type`` (str): Type of simulator to use
    
    - ``shots`` (int): Number of measurement shots to perform

**Example:**

.. code-block:: python

   backend_config = {
       "backend_name": "qiskit",
       "backend_options": {
           "simulator_type": "aer_simulator",
           "shots": 1024,
       },
   }
   qumat = QuMat(backend_config)

**Raises:**

- **ImportError**: If the specified backend module cannot be imported.

- **KeyError**: If required configuration keys are missing.

## Circuit Creation

### `create_empty_circuit(self, num_qubits)`

Create an empty quantum circuit with a specified number of qubits.

**Parameters:**

- **num_qubits** (int or None, optional): Number of qubits in the quantum circuit. Must be a non-negative integer if provided. Defaults to None.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)

**Note:**

This method must be called before applying any gates or executing operations on the circuit.

## Basic Quantum Gates

### `apply_not_gate(self, qubit_index)`

Apply a NOT gate (Pauli-X gate) to a specified qubit.

The NOT gate flips the state of a qubit: |0⟩ → |1⟩ and |1⟩ → |0⟩.

**Parameters:**

- **qubit_index** (int): Index of the qubit to which the gate is applied.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_not_gate(qubit_index=0)

### `apply_hadamard_gate(self, qubit_index)`

Apply a Hadamard gate to a specified qubit.

The Hadamard gate creates a superposition state: |0⟩ → (|0⟩ + |1⟩) / √2.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_hadamard_gate(qubit_index=0)

### `apply_cnot_gate(self, control_qubit_index, target_qubit_index)`

Apply a Controlled-NOT (CNOT) gate between two qubits.

The CNOT gate entangles two qubits. If the control qubit is |1⟩, it flips the target qubit.

**Parameters:**

- **control_qubit_index** (int): Index of the control qubit.

- **target_qubit_index** (int): Index of the target qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_hadamard_gate(0)
   qumat.apply_cnot_gate(control_qubit_index=0, target_qubit_index=1)

### `apply_toffoli_gate(self, control_qubit_index1, control_qubit_index2, target_qubit_index)`

Apply a Toffoli gate (CCX gate) to three qubits.

The Toffoli gate acts as a controlled-controlled-X gate. It applies the X gate to the target qubit only if both control qubits are |1⟩.

**Parameters:**

- **control_qubit_index1** (int): Index of the first control qubit.

- **control_qubit_index2** (int): Index of the second control qubit.

- **target_qubit_index** (int): Index of the target qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)
   qumat.apply_toffoli_gate(0, 1, 2)

### `apply_swap_gate(self, qubit_index1, qubit_index2)`

Swap the states of two qubits.

The SWAP gate exchanges the quantum states of two qubits.

**Parameters:**

- **qubit_index1** (int): Index of the first qubit.

- **qubit_index2** (int): Index of the second qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_swap_gate(qubit_index1=0, qubit_index2=1)

### `apply_cswap_gate(self, control_qubit_index, target_qubit_index1, target_qubit_index2)`

Apply a Controlled-SWAP (CSWAP) gate, also known as a Fredkin gate.

The CSWAP gate conditionally swaps two target qubits based on the state of a control qubit. If the control qubit is |1⟩, it swaps the two target qubits; otherwise, it does nothing.

**Parameters:**

- **control_qubit_index** (int): Index of the control qubit.

- **target_qubit_index1** (int): Index of the first target qubit.

- **target_qubit_index2** (int): Index of the second target qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)
   qumat.apply_cswap_gate(control_qubit_index=0, target_qubit_index1=1, target_qubit_index2=2)

**Note:**

This gate is commonly used in quantum algorithms such as the swap test for measuring overlap between quantum states.

## Pauli Gates

### `apply_pauli_x_gate(self, qubit_index)`

Apply a Pauli-X gate to a specified qubit.

The Pauli-X gate is equivalent to the NOT gate and flips the qubit state.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_pauli_x_gate(qubit_index=0)

### `apply_pauli_y_gate(self, qubit_index)`

Apply a Pauli-Y gate to a specified qubit.

The Pauli-Y gate introduces complex phase shifts along with a bit-flip operation.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_pauli_y_gate(qubit_index=0)

### `apply_pauli_z_gate(self, qubit_index)`

Apply a Pauli-Z gate to a specified qubit.

The Pauli-Z gate introduces a phase flip without changing the qubit's state: |0⟩ → |0⟩ and |1⟩ → -|1⟩.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_pauli_z_gate(qubit_index=0)

## Rotation Gates

### `apply_rx_gate(self, qubit_index, angle)`

Apply a rotation around the X-axis to a specified qubit.

The Rx gate rotates the qubit state around the X-axis of the Bloch sphere by the specified angle.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

- **angle** (float or str): Rotation angle in radians. Can be:
  
  - A float value for a static rotation angle.
  
  - A string parameter name for use in parameterized circuits (e.g., ``"theta"``).

**Example:**

.. code-block:: python

   import numpy as np
   
   # Static angle
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_rx_gate(qubit_index=0, angle=np.pi / 2)
   
   # Parameterized angle
   qumat.apply_rx_gate(qubit_index=0, angle="theta")

**Note:**

When using a string parameter name, the parameter will be automatically registered and can be bound later using ``bind_parameters()`` or ``execute_circuit(parameter_values={...})``.

### `apply_ry_gate(self, qubit_index, angle)`

Apply a rotation around the Y-axis to a specified qubit.

The Ry gate rotates the qubit state around the Y-axis of the Bloch sphere by the specified angle.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

- **angle** (float or str): Rotation angle in radians. Can be:
  
  - A float value for a static rotation angle.
  
  - A string parameter name for use in parameterized circuits (e.g., ``"phi"``).

**Example:**

.. code-block:: python

   import numpy as np
   
   # Static angle
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_ry_gate(qubit_index=0, angle=np.pi / 4)
   
   # Parameterized angle
   qumat.apply_ry_gate(qubit_index=0, angle="phi")

**Note:**

When using a string parameter name, the parameter will be automatically registered and can be bound later using ``bind_parameters()`` or ``execute_circuit(parameter_values={...})``.

### `apply_rz_gate(self, qubit_index, angle)`

Apply a rotation around the Z-axis to a specified qubit.

The Rz gate rotates the qubit state around the Z-axis of the Bloch sphere by the specified angle, modifying the phase.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

- **angle** (float or str): Rotation angle in radians. Can be:
  
  - A float value for a static rotation angle.
  
  - A string parameter name for use in parameterized circuits (e.g., ``"lambda"``).

**Example:**

.. code-block:: python

   import numpy as np
   
   # Static angle
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_rz_gate(qubit_index=0, angle=np.pi)
   
   # Parameterized angle
   qumat.apply_rz_gate(qubit_index=0, angle="lambda")

**Note:**

When using a string parameter name, the parameter will be automatically registered and can be bound later using ``bind_parameters()`` or ``execute_circuit(parameter_values={...})``.

## Universal Gate

### `apply_u_gate(self, qubit_index, theta, phi, lambd)`

Apply a universal single-qubit gate (U gate) to a specified qubit.

The U gate is a parameterized gate that can represent any single-qubit operation. It is defined by three Euler angles: theta, phi, and lambda.

**Parameters:**

- **qubit_index** (int): Index of the qubit.

- **theta** (float): First rotation angle in radians.

- **phi** (float): Second rotation angle in radians.

- **lambd** (float): Third rotation angle in radians.

**Example:**

.. code-block:: python

   import numpy as np
   
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_u_gate(qubit_index=0, theta=np.pi/2, phi=np.pi/4, lambd=np.pi/3)

**Note:**

The U gate can be decomposed into rotations: U(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ). Different backends may implement this gate differently, but the mathematical effect is equivalent.

## Circuit Execution

### `execute_circuit(self, parameter_values=None)`

Execute the quantum circuit and retrieve measurement results.

This method runs the entire quantum circuit, performs measurements, and returns the results. For parameterized circuits, parameter values can be provided to bind before execution.

**Parameters:**

- **parameter_values** (dict, optional): Dictionary mapping parameter names to numerical values. Keys should be strings matching parameter names used in rotation gates. If provided, parameters are bound before execution.

**Returns:**

The return format depends on the backend:

- **Qiskit/Braket**: Returns a dictionary with string keys representing measurement outcomes (e.g., ``{"000": 512, "001": 512}``) and integer values representing the count of each outcome.

- **Cirq**: Returns a list of dictionaries with integer keys representing measurement outcomes (e.g., ``[{0: 512, 1: 512}]``) and integer values representing the count of each outcome.

**Example:**

.. code-block:: python

   # Execute a simple circuit
   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_hadamard_gate(0)
   qumat.apply_cnot_gate(0, 1)
   results = qumat.execute_circuit()
   print(results)  # e.g., {"00": 512, "11": 512}
   
   # Execute a parameterized circuit
   import numpy as np
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_rx_gate(0, angle="theta")
   results = qumat.execute_circuit(parameter_values={"theta": np.pi / 2})
   print(results)

**Note:**

- The circuit must be initialized using ``create_empty_circuit()`` before execution.

- For circuits with zero qubits, the method returns a special result format indicating all shots measured the empty state.

- The number of shots is determined by the ``shots`` parameter in ``backend_options`` during initialization.

## Parameter Binding

### `bind_parameters(self, parameter_values)`

Bind numerical values to parameters in the quantum circuit.

This method updates the parameter values for parameterized gates in the circuit. Parameters must have been previously registered (typically by using string parameter names in rotation gates).

**Parameters:**

- **parameter_values** (dict): Dictionary mapping parameter names (strings) to numerical values (floats). Only parameters that have been registered in the circuit will be updated.

**Example:**

.. code-block:: python

   import numpy as np
   
   # Create a parameterized circuit
   qumat.create_empty_circuit(num_qubits=1)
   qumat.apply_rx_gate(0, angle="theta")
   qumat.apply_ry_gate(0, angle="phi")
   
   # Bind parameter values
   qumat.bind_parameters({"theta": np.pi / 2, "phi": np.pi / 4})
   
   # Execute with bound parameters
   results = qumat.execute_circuit()

**Note:**

- Parameters are automatically registered when using string parameter names in rotation gates (``apply_rx_gate()``, ``apply_ry_gate()``, ``apply_rz_gate()``).

- If a parameter name in ``parameter_values`` is not registered in the circuit, it will be silently ignored.

- This method is useful for optimization loops where parameters are adjusted iteratively.

## Circuit Visualization

### `draw(self)`

Visualize the quantum circuit.

This method returns a visual representation of the quantum circuit, which can be printed or displayed. The exact format depends on the backend implementation.

**Returns:**

The return type depends on the backend:

- **Qiskit**: Returns a text or image representation of the circuit.

- **Cirq**: Returns a text representation of the circuit.

- **Braket**: Returns a text representation of the circuit object.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_hadamard_gate(0)
   qumat.apply_cnot_gate(0, 1)
   circuit_diagram = qumat.draw()
   print(circuit_diagram)

**Note:**

- The circuit must be initialized using ``create_empty_circuit()`` before drawing.

- The visualization format is backend-specific and may vary in appearance.

## Quantum State Measurement

### `swap_test(self, ancilla_qubit, qubit1, qubit2)`

Implements the swap test circuit for measuring overlap between two quantum states.

The swap test is a quantum algorithm that measures the inner product (overlap) between two quantum states. It uses an ancilla qubit to perform the measurement. The probability of measuring the ancilla qubit in state |0⟩ is related to the overlap as: P(0) = (1 + |⟨ψ|φ⟩|²) / 2.

**Parameters:**

- **ancilla_qubit** (int): Index of the ancilla qubit. This qubit should be initialized to |0⟩.

- **qubit1** (int): Index of the first qubit containing state |ψ⟩.

- **qubit2** (int): Index of the second qubit containing state |φ⟩.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)
   # Prepare states on qubit1 and qubit2 (not shown)
   qumat.swap_test(ancilla_qubit=0, qubit1=1, qubit2=2)
   results = qumat.execute_circuit()

**Note:**

- The swap test circuit consists of applying a Hadamard gate to the ancilla, a controlled-SWAP gate, and another Hadamard gate to the ancilla.

- After executing the circuit, the measurement results can be used to calculate the overlap between the two states.

### `measure_overlap(self, qubit1, qubit2, ancilla_qubit=0)`

Measure the overlap (fidelity) between two quantum states using the swap test.

This method performs a complete swap test and returns the squared overlap |⟨ψ|φ⟩|², which represents the fidelity between the two quantum states.

**Parameters:**

- **qubit1** (int): Index of the first qubit containing state |ψ⟩.

- **qubit2** (int): Index of the second qubit containing state |φ⟩.

- **ancilla_qubit** (int, optional): Index of the ancilla qubit. Defaults to 0. This qubit should be initialized to |0⟩.

**Returns:**

- **float**: The squared overlap |⟨ψ|φ⟩|² between the two states (fidelity), a value between 0.0 and 1.0.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)
   # Prepare |0⟩ on qubit1 and |0⟩ on qubit2 (default state)
   overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
   print(f"Overlap: {overlap}")  # Should be close to 1.0 for identical states

**Note:**

- This method internally calls ``swap_test()`` and ``execute_circuit()``, then calculates the overlap from the measurement results.

- For certain states (especially identical excited states), global phase effects may cause the ancilla to measure predominantly |1⟩ instead of |0⟩. This method handles both cases by taking the measurement probability closer to 1.

- The returned value is always clamped between 0.0 and 1.0.

### `calculate_prob_zero(self, results, ancilla_qubit)`

Calculate the probability of measuring the ancilla qubit in the |0⟩ state.

This method processes measurement results from ``execute_circuit()`` to determine the probability that the ancilla qubit was measured in the |0⟩ state. It delegates to backend-specific implementations to handle different result formats.

**Parameters:**

- **results**: Measurement results from ``execute_circuit()``. The format depends on the backend:
  
  - **Qiskit/Braket**: Dictionary with string keys (e.g., ``{"000": 512, "001": 512}``).
  
  - **Cirq**: List of dictionaries with integer keys (e.g., ``[{0: 512, 1: 512}]``).

- **ancilla_qubit** (int): Index of the ancilla qubit.

**Returns:**

- **float**: Probability of measuring the ancilla qubit in |0⟩ state, a value between 0.0 and 1.0.

**Example:**

.. code-block:: python

   qumat.create_empty_circuit(num_qubits=3)
   qumat.swap_test(ancilla_qubit=0, qubit1=1, qubit2=2)
   results = qumat.execute_circuit()
   prob_zero = qumat.calculate_prob_zero(results, ancilla_qubit=0)
   print(f"Probability of measuring |0⟩: {prob_zero}")

**Note:**

- This method is primarily used internally by ``measure_overlap()`` but can be called directly if needed.

- The method handles different result formats from different backends automatically.

## Testing and Debugging

### `get_final_state_vector(self)`

Get the final state vector of the quantum circuit.

This method is primarily intended for use in testing and debugging. It returns the complete quantum state vector after all gates have been applied but before measurement.

**Returns:**

- **numpy.ndarray**: The final state vector as a complex-valued array. The length of the array is 2^n where n is the number of qubits.

**Example:**

.. code-block:: python

   import numpy as np
   
   qumat.create_empty_circuit(num_qubits=2)
   qumat.apply_hadamard_gate(0)
   qumat.apply_cnot_gate(0, 1)
   state_vector = qumat.get_final_state_vector()
   print(f"State vector: {state_vector}")

**Note:**

- This method is marked as a placeholder for use in the testing suite.

- The state vector represents the quantum state in the computational basis.

- For large numbers of qubits, the state vector can be very large (2^n complex numbers).

- This method does not perform measurements; it returns the full quantum state.
