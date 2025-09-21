import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. Euler's Method
# ==================================================
def euler_method(f, x0, y0, h, n):
    x, y = x0, y0
    results = [(x, y)]
    for i in range(n):
        y = y + h * f(x, y)
        x = x + h
        results.append((x, y))
    return results

# ==================================================
# 2. Milne's Method + Adams-Bashforth 4th Order
# ==================================================
def milne_method(f, x0, y0, h, n):
    def rk4(f, x0, y0, h):
        k1 = f(x0, y0)
        k2 = f(x0 + h/2, y0 + h*k1/2)
        k3 = f(x0 + h/2, y0 + h*k2/2)
        k4 = f(x0 + h, y0 + h*k3)
        return y0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    xs = [x0]
    ys = [y0]

    for i in range(3):
        y_new = rk4(f, xs[-1], ys[-1], h)
        xs.append(xs[-1] + h)
        ys.append(y_new)

    for i in range(3, n):
        xp = xs[-1] + h
        yp = ys[-4] + (4*h/3) * (2*f(xs[-3], ys[-3]) - f(xs[-2], ys[-2]) + 2*f(xs[-1], ys[-1]))
        yc = ys[-2] + (h/3) * (f(xp, yp) + 4*f(xs[-1], ys[-1]) + f(xs[-2], ys[-2]))
        xs.append(xp)
        ys.append(yc)

    return list(zip(xs, ys))


def adams_bashforth_4(f, x0, y0, h, n):
    def rk4(f, x0, y0, h):
        k1 = f(x0, y0)
        k2 = f(x0 + h/2, y0 + h*k1/2)
        k3 = f(x0 + h/2, y0 + h*k2/2)
        k4 = f(x0 + h, y0 + h*k3)
        return y0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    xs = [x0]
    ys = [y0]

    for i in range(3):
        y_new = rk4(f, xs[-1], ys[-1], h)
        xs.append(xs[-1] + h)
        ys.append(y_new)

    for i in range(3, n):
        y_new = ys[-1] + (h/24)*(55*f(xs[-1], ys[-1]) - 59*f(xs[-2], ys[-2]) + 37*f(xs[-3], ys[-3]) - 9*f(xs[-4], ys[-4]))
        xs.append(xs[-1] + h)
        ys.append(y_new)

    return list(zip(xs, ys))

# ==================================================
# 3. Vector Space Check (10 conditions)
# ==================================================
def check_vector_space(vectors):
    results = {}
    zero_vec = np.zeros_like(vectors[0])

    results['closure_addition'] = all(
        any(np.array_equal(v1+v2, v) for v in vectors)
        for v1 in vectors for v2 in vectors
    )

    results['closure_scalar'] = all(
        any(np.array_equal(2*v, vec) for vec in vectors)
        for v in vectors
    )

    results['commutativity'] = all(
        np.array_equal(v1+v2, v2+v1) for v1 in vectors for v2 in vectors
    )

    results['associativity'] = all(
        np.array_equal((v1+v2)+v3, v1+(v2+v3)) for v1 in vectors for v2 in vectors for v3 in vectors
    )

    results['additive_identity'] = any(
        np.array_equal(v + zero_vec, v) for v in vectors
    )

    results['additive_inverse'] = all(
        any(np.array_equal(-v, vec) for vec in vectors) for v in vectors
    )

    results['distributivity1'] = all(
        np.array_equal(2*(v1+v2), 2*v1 + 2*v2) for v1 in vectors for v2 in vectors
    )

    results['distributivity2'] = all(
        np.array_equal((2+3)*v, 2*v + 3*v) for v in vectors
    )

    results['scalar_compatibility'] = all(
        np.array_equal(2*(3*v), (2*3)*v) for v in vectors
    )

    results['scalar_identity'] = all(
        np.array_equal(1*v, v) for v in vectors
    )

    return results

# ==================================================
# Streamlit App
# ==================================================

def main():
    st.title("📘 Online Numerical Methods Calculator")

    tabs = st.tabs(["Euler Method", "Milne/Adams-Bashforth", "Vector Space Check"])

    # ---------------------- Euler ----------------------
    with tabs[0]:
        st.header("Euler's Method")
        func_str = st.text_input("Enter f(x,y):", "x + y")
        x0 = st.number_input("Initial x0", value=0.0)
        y0 = st.number_input("Initial y0", value=1.0)
        h = st.number_input("Step size h", value=0.1)
        n = st.number_input("Number of steps", value=10, step=1)

        def f(x, y):
            return eval(func_str, {'x': x, 'y': y, 'np': np})

        if st.button("Run Euler Method", key='euler'):
            results = euler_method(f, x0, y0, h, n)
            st.write("Results:")
            st.table(results)

            xs, ys = zip(*results)
            st.line_chart({'x': xs, 'y': ys})

    # -------------------- Milne/Adams --------------------
    with tabs[1]:
        st.header("Milne's & Adams-Bashforth 4th Order")
        func_str2 = st.text_input("Enter f(x,y):", "x + y", key='milne_func')
        x0_2 = st.number_input("Initial x0", value=0.0, key='milne_x0')
        y0_2 = st.number_input("Initial y0", value=1.0, key='milne_y0')
        h2 = st.number_input("Step size h", value=0.1, key='milne_h')
        n2 = st.number_input("Number of steps", value=10, step=1, key='milne_n')
        method_choice = st.selectbox("Choose Method", ["Milne", "Adams-Bashforth"], key='method_choice')

        def f2(x, y):
            return eval(func_str2, {'x': x, 'y': y, 'np': np})

        if st.button("Run Method", key='milne_adams'):
            if method_choice == 'Milne':
                results2 = milne_method(f2, x0_2, y0_2, h2, n2)
            else:
                results2 = adams_bashforth_4(f2, x0_2, y0_2, h2, n2)
            st.write("Results:")
            st.table(results2)

            xs2, ys2 = zip(*results2)
            st.line_chart({'x': xs2, 'y': ys2})

    # ------------------- Vector Space -------------------
    with tabs[2]:
        st.header("Vector Space 10 Conditions Check")
        st.write("Example: vectors = [[0,0],[1,0],[0,1],[1,1]]")
        vectors_input = st.text_area("Enter vectors as list of lists", "[[0,0],[1,0],[0,1],[1,1]]")

        if st.button("Check Vector Space", key='vs'):
            try:
                vectors_list = eval(vectors_input)
                vectors_np = [np.array(v) for v in vectors_list]
                res_vs = check_vector_space(vectors_np)
                st.write(res_vs)
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
