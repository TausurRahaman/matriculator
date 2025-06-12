import streamlit as st
from core.matrix import Matrix
from collections import defaultdict
import re
from sympy import Matrix as SymMatrix
from sympy import nsimplify, N
import numbers

def format_number(val, tol=1e-10):
    # Convert to SymPy if needed
    from sympy import sympify, I

    try:
        val = sympify(val)
        real, imag = val.as_real_imag()
    except Exception:
        # Fallback if sympify fails
        real, imag = val.real, val.imag

    real = float(real)
    imag = float(imag)

    # Round tiny values to 0
    if abs(real) < tol:
        real = 0
    if abs(imag) < tol:
        imag = 0

    # Format output
    if imag == 0:
        return str(int(real)) if real == int(real) else f"{real:.5f}"
    elif real == 0:
        return f"{imag:.5f} 𝒊" if abs(imag) != 1 else ("𝒊" if imag > 0 else " -𝒊")
    else:
        sign = '+' if imag > 0 else '-'
        imag_abs = abs(imag)
        imag_str = f"{imag_abs:.5f} 𝒊" if imag_abs != 1 else " 𝒊"
        real_str = str(int(real)) if real == int(real) else f"{real:.5f}"
        return f"{real_str} {sign} {imag_str}"



def matrix_input(label, rows, cols, key_prefix):
    st.write(f"### {label}")
    data = []
    for i in range(rows):
        row = []
        cols_row = st.columns(cols)
        for j in range(cols):
            val = cols_row[j].number_input(
                f"{label}[{i+1},{j+1}]",
                key=f"{key_prefix}_{i}_{j}",
                value=0
            )
            row.append(val)
        data.append(row)
    return Matrix(rows, cols, data)

def display_matrix(matrix):
    st.subheader("Result Matrix")

    html = """
    <style>
    .matrix-grid {
        display: inline-block;
        margin-top: 16px;
        padding: 8px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .matrix-row {
        display: flex;
    }
    .matrix-cell {
        min-width: 60px;
        min-height: 40px;
        margin: 4px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    <div class="matrix-grid">
    """

    for row in matrix.to_list():
        html += '<div class="matrix-row">'
        for val in row:
            html += f'<div class="matrix-cell">{format_number(val)}</div>'
        html += '</div>'

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)



# Polynomial evaluation on matrix A given coefficients
def evaluate_polynomial(matrix, coeffs):
    # Polynomial: a_n * A^n + ... + a_1 * A + a_0 * I
    degree = len(coeffs) - 1
    result = Matrix.identity(matrix.row) * coeffs[0]  # a_0 * I
    for power in range(1, degree + 1):
        term = matrix ** power
        term = term * coeffs[power]
        result = result + term
    return result


# Solve linear polynomial for unknown matrix X: a_1 X + a_0 I = B
def solve_linear_polynomial_matrix(coeffs, B):
    # coeffs: [a_0, a_1], degree must be 1
    if len(coeffs) != 2:
        raise ValueError("Only linear polynomials are supported for solving.")
    a0, a1 = coeffs

    if a1 == 0:
        raise ValueError("Coefficient for X (a_1) cannot be zero.")

    I = Matrix.identity(B.row)
    term = I * a0
    right_side = B - term
    X = right_side * (1 / a1)
    return X

# Initialize session state for number of matrices
if "num_matrices" not in st.session_state:
    st.session_state.num_matrices = 1

st.title("🧮 MatriCulator")

st.sidebar.subheader("Matrices")

# Buttons to add/remove matrices
col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add Matrix"):
    st.session_state.num_matrices += 1
if col2.button("➖ Remove"):
    if st.session_state.num_matrices > 1:
        st.session_state.num_matrices -= 1

num = st.session_state.num_matrices
st.sidebar.write(f"Total Matrices: {num}")

# Select operation
operation = st.selectbox(
    "Select Operation",
    [
        "Add",
        "Subtract",
        "Multiply",
        "Transpose",
        "Determinant",
        "Inverse",
        "Power",
        "Rank",
        "Polynomial Evaluate",
        "Polynomial Solve",
        "Eigenvalues and Eigenvectors",
        "Solve Linear System (Ax = b)",
        "Singular Value Decomposition (SVD)"
    ]
)

if operation == "Polynomial Solve":
    st.markdown("### Solve equation: `a·X + c·I = B`  &nbsp;&nbsp;&nbsp; where `X` is the unknown matrix")
    st.markdown("<small><i>Note: Only linear matrix equations can be solved (degree 1 polynomial).</i></small>", unsafe_allow_html=True)



# For single-matrix operation, let user pick which matrix to operate on
single_matrix_ops = ["Transpose", "Determinant", "Inverse", "Power","Rank"]
selected_index = 0  # default value

if operation in single_matrix_ops:
    selected_index = st.selectbox(
    "Select matrix for operation",
    options=list(range(st.session_state.num_matrices)),
    format_func=lambda x: f"Matrix {x+1}"
)
    
# For Power operation, show power input (only when power selected)
power_value = None
if operation == "Power":
    power_value = st.number_input("Enter power (integer ≥ 0)", min_value=0, value=2, step=1)

# Gather all matrices
matrices = []
shapes = []

for idx in range(num):
    st.sidebar.markdown(f"**Matrix {idx+1} Dimensions**")
    rows = st.sidebar.number_input(f"Rows {idx+1}", min_value=1, value=2, key=f"rows_{idx}")
    cols = st.sidebar.number_input(f"Cols {idx+1}", min_value=1, value=2, key=f"cols_{idx}")
    shapes.append((rows, cols))

    mat = matrix_input(f"Matrix {idx+1}", rows, cols, key_prefix=f"m_{idx}")
    matrices.append(mat)


if operation == "Polynomial Evaluate":
    if len(matrices) == 0:
        st.error("Please declare at least one matrix.")
    else:
        selected_label = st.selectbox(
            "Select a matrix to evaluate:",
            [f"Matrix {i+1}" for i in range(len(matrices))],
            key="poly_eval_matrix_select"
        )
        selected_index = int(selected_label.split()[-1]) - 1
        A = matrices[selected_index]

        if A.row != A.col:
            st.error("Polynomial evaluation requires a square matrix.")
        else:
            expression = st.text_input(
                "Enter polynomial expression (e.g., 5A^2 + 2A + 1):",
                key="poly_expression_input"
            )


if operation == "Polynomial Solve":
    if len(matrices) == 0:
        st.error("Please declare at least one matrix.")
    else:
        selected_label = st.selectbox(
            "Select matrix B to solve with:",
            [f"Matrix {i+1}" for i in range(len(matrices))],
            key="poly_solve_matrix_select"
        )
        selected_index = int(selected_label.split()[-1]) - 1
        B = matrices[selected_index]

        if B.row != B.col:
            st.error("Polynomial solve requires a square matrix.")
        else:
            a0 = st.number_input("Enter constant term (a₀):", key="poly_solve_a0", value=1.0)
            a1 = st.number_input("Enter coefficient of X (a₁):", key="poly_solve_a1", value=1.0)


if operation == "Eigenvalues and Eigenvectors":
    selected_index = st.selectbox(
        "Select a matrix to analyze:",
        options=list(range(st.session_state.num_matrices)),
        format_func=lambda x: f"Matrix {x+1}",
        key="eig_matrix_select"
    )


if operation == "Solve Linear System (Ax = b)":
    mat = matrices[selected_index]

    b_data = []
    valid_input = True
    b_matrix = None

    if not mat.isSquareMatrix():
        st.warning("Matrix A must be square to solve Ax = b.")
    else:
        st.subheader("Enter vector b:")
        for i in range(mat.row):
            val = st.text_input(f"b[{i}][0]", value="0.0", key=f"b_input_{i}")
            try:
                b_data.append([complex(eval(val))])
            except Exception:
                st.error(f"Invalid input at b[{i}][0].")
                valid_input = False
                break

        if valid_input:
            b_matrix = Matrix(mat.row, 1, b_data)


if operation == "Singular Value Decomposition (SVD)":
    selected_index = st.selectbox(
        "Select a matrix to decompose:",
        options=list(range(st.session_state.num_matrices)),
        format_func=lambda x: f"Matrix {x+1}",
        key="svd_matrix_select"
    )



# Validation logic and calculation
if st.button("Calculate"):
    
    try:
        result = None

        if operation in ["Add", "Subtract"]:
            # Ensure all shapes match
            if not all(shape == shapes[0] for shape in shapes):
                st.error("All matrices must have the same dimensions for addition/subtraction.")
            else:
                result = matrices[0]
                for mat in matrices[1:]:
                    result = result + mat if operation == "Add" else result - mat

        elif operation == "Multiply":
            # Ensure chain multiplication compatibility
            compatible = True
            for i in range(len(shapes)-1):
                if shapes[i][1] != shapes[i+1][0]:
                    compatible = False
                    break
            if not compatible:
                st.error("For multiplication, A.cols must equal B.rows for all adjacent matrices.")
            else:
                result = matrices[0]
                for mat in matrices[1:]:
                    result = result * mat

        elif operation in single_matrix_ops:
            mat = matrices[selected_index]

            if operation == "Transpose":
                result = mat.transpose()

            elif operation == "Determinant":
                if mat.row != mat.col:
                    st.error("Matrix must be square to calculate determinant.")
                else:
                    det = mat.determinant()
                    st.success(f"Determinant: {det}")
                    result = None  # No matrix to display

            elif operation == "Inverse":
                if mat.row != mat.col:
                    st.error("Matrix must be square to calculate inverse.")
                else:
                    result = mat.inverse()

            elif operation == "Power":
                # power_value is already from input above
                result = mat ** power_value

            elif operation == "Rank":
                mat = matrices[selected_index]
                # Use sympy Matrix for rank calculation
                sym_mat = SymMatrix(mat.to_list())
                rank = sym_mat.rank()
                st.success(f"Rank of Matrix {selected_index+1}: {rank}")
                result = None  # No matrix output needed here

        elif operation == "Polynomial Evaluate":
            A = matrices[selected_index]
            if A.row != A.col:
                st.error("Matrix must be square for polynomial evaluation.")
            else:
                expr = expression.replace(" ", "")
                terms = re.findall(r'([+-]?\d*)A(?:\^(\d+))?|([+-]?\d+)', expr)

                degree_coeffs = defaultdict(int)
                for coeff, power, constant in terms:
                    if constant:
                        degree_coeffs[0] += int(constant)
                    else:
                        c = int(coeff) if coeff not in ["", "+", "-"] else 1 if coeff in ["", "+"] else -1
                        p = int(power) if power else 1
                        degree_coeffs[p] += c

                max_degree = max(degree_coeffs.keys())
                coeffs = [degree_coeffs.get(i, 0) for i in range(max_degree + 1)]

                result = evaluate_polynomial(A, coeffs)

        elif operation == "Polynomial Solve":
            if a1 == 0:
                st.error("Coefficient of X (a₁) cannot be zero.")
            else:
                I = Matrix.identity(B.row)
                term = I * a0
                right_side = B - term
                result = right_side * (1 / a1)

        elif operation == "Eigenvalues and Eigenvectors":
            mat = matrices[selected_index]

            if mat.row != mat.col:
                st.error("Matrix must be square to compute eigenvalues.")
            else:
                eigenvals = mat.eigenvalues()
                eigenvecs = mat.eigenvectors()

                st.success("Eigenvalues:")
                for val, mult in eigenvals.items():
                    st.write(f"{format_number(val)} (multiplicity: {mult})")



                st.subheader("Eigenvectors:")
                for val, mult, vecs in eigenvecs:
                    st.markdown(f"**Eigenvalue:** {format_number(val)} (multiplicity: {mult})")
                    for vec in vecs:
                        py_matrix = [[complex(x.evalf())] for x in vec]
                        custom_matrix = Matrix(len(py_matrix), 1, py_matrix)
                        display_matrix(custom_matrix)

        elif operation == "Solve Linear System (Ax = b)":
            mat = matrices[selected_index]
            if not b_matrix:
                st.error("Invalid or incomplete input for b vector.")
            else:
                x = mat.solve_linear_system(b_matrix)
                st.success("Solution vector x:")
                display_matrix(x)

        elif operation == "Singular Value Decomposition (SVD)":
            mat = matrices[selected_index]
            sym_mat = SymMatrix(mat.to_list())
            U, Sigma, V = sym_mat.singular_value_decomposition()

            # Convert to your custom Matrix class
            U_mat = Matrix(U.rows, U.cols, U.tolist())
            # Sigma is a diagonal matrix, shape m x n
            Sigma_mat = Matrix(Sigma.rows, Sigma.cols, Sigma.tolist())
            V_mat = Matrix(V.rows, V.cols, V.tolist())

            st.success("Singular Value Decomposition Result:")
            st.markdown("**Matrix U:**")
            display_matrix(U_mat)
            st.markdown("**Matrix Σ (Sigma):**")
            display_matrix(Sigma_mat)
            st.markdown("**Matrix Vᵀ:**")
            display_matrix(V_mat)



        # Display
        if result:
            if operation == "Polynomial Solve": st.success(f"Solved Matrix X:")
            else : st.success(f"{operation} result:")
            display_matrix(result)

    except Exception as e:
        st.error(f"Error: {e}")


# Footer
st.markdown(
    """
    <style>
    .footer {
        margin-top: 50px;
        padding: 12px 0;
        text-align: center;
        font-size: 16px;
        color: white;
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        user-select: none;
    }
    .footer span {
        color: #00bcd4;
        font-weight: bold;
    }
    .footer a {
        margin: 0 10px;
        color: white;
        text-decoration: none;
        vertical-align: middle;
        display: inline-block;
        width: 32px;
        height: 32px;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #00bcd4;
    }
    .footer svg {
        fill: currentColor;
        width: 20px;
        height: 20px;
        vertical-align: middle;
    }
    </style>

    <div class="footer">
        Created by <span>TAUSUR</span>&nbsp;&nbsp;&nbsp; ||
        <a href="https://github.com/TausurRahaman" target="_blank">
            <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <title>GitHub</title>
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 
                0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.09-.745.083-.73.083-.73 
                1.205.084 1.84 1.238 1.84 1.238 1.07 1.835 2.807 1.305 3.492.997.107-.776.42-1.305.763-1.605-2.665-.3-5.467-1.335-5.467-5.93 
                0-1.31.468-2.38 1.235-3.22-.123-.303-.535-1.523.117-3.176 0 0 1.008-.322 3.3 1.23a11.52 11.52 0 013.003-.404c1.02.005 2.045.138 
                3.003.404 2.29-1.552 3.296-1.23 3.296-1.23.653 1.653.242 2.873.12 3.176.77.84 1.233 1.91 1.233 3.22 0 4.61-2.807 5.625-5.48 
                5.92.43.37.823 1.103.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .32.217.694.825.576C20.565 21.795 24 17.296 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
        </a>
        <a href="https://tausur-rahaman.vercel.app/" target="_blank">🔗</a>
        <a href="mailto:tausur345@gmail.com">✉️</a>
    </div>
    """,
    unsafe_allow_html=True
)