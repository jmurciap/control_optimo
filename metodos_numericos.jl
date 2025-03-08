### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3b7c3a68-51d2-48dd-ac34-3ff43e0f30fd
begin
	using WGLMakie
	using LinearAlgebra
	using ForwardDiff
	using Plots
	using PlutoUI
end

# ╔═╡ 8d88f02c-3c87-4367-af35-171c61f0a087
md"""
Juan Diego Murcia Porras, 2025
"""

# ╔═╡ cb2cef50-dcd1-11ef-18c8-035ede97281d
md"""
# Métodos de control óptimo
"""

# ╔═╡ adf66ca2-3f1f-4954-b2d0-1bd365e553b6
md"""
En este cuaderno, se exploran tres distintos métodos para resolver problemas de control óptimo, tomadno como prueba el problema:

$\min_u \frac{1}{2}\int_{0}^1x(t)^2+u(t)^2~dt,$
sujeto a la dinámica

$\begin{cases}
x'(t)=-x(t)+u(t)\\
x(0)=1
\end{cases}$

Es importante resaltar que se implementa este problema a manera de ilustración, pues este problema se enmarca dentro de los problemas de tipo regulador cuaddrático lineal, y su solución explícita puede ser hallada.
"""

# ╔═╡ 57be2c9b-37bb-4759-b2a7-a565669b524a
md"""
## Programación Dinámica Diferencial (DDP)
"""

# ╔═╡ cca1cb95-cbda-4bd7-a33e-23c73b46e945
md"""
Considere el problema (independiente del tiempo)

$\min_u \int_{t_0}^{t_f} L(x,u)dt+\phi(x(t_f)),~x\in \mathbb{R}^n, u\in \mathbb{R}^m$
sujeto a la dinámica

$\begin{cases}\dot{x}=f(x,u)\\
x(t_0)=x_0\end{cases}.$

Podemos realizar una discretización del anterior problema tomando $X=[x(t_0),x(t_1),\ldots, x(t_N)]:=[x_0,x_1,\ldots, x_N]$ y $U=[u_1,\ldots, u_{N-1}].$ Asumiendo una aproximación de la dinámica $g(x,u)$ que cumple $x_{k+1}\approx g(x_k,u_k)$, debemos encontrar

$\min_{U}\sum_{i=1}
^{N-1}L(x_i,u_i)+\phi(x_N)$
sujeto a la dinámica

$\begin{cases}
x_{k+1}=g(x_k,u_k),~k=0,1,2,\ldots, N-1\\
x_0=x_0.
\end{cases}$
El anterior es un problema de optimización regular con una variable de estado $U$ ($Nm$  dimensional). Este problema se puede intentar resolver directamente, pero no resulta muy eficiente (sobre todo porque el cálculo de los gradientes no es muy eficiente), y no explota la estructura del problema.

Definimos el costo óptimo que resta en $V_{k}(x)$ (definida en inglés como 'cost-to-go' function) como el costo mínimo que resta por asumir desde un punto $x$ en el tiempo $t_k$. La ecuación de Bellman enuncia que

$V_k(x)=\min_{u}\{L(x,u)+V_{k+1}(g(x,u))\}:=\min_{u}\{S_{k}(x,u)\},~k=0,1,\ldots, N-1.$
con condición final $V_{N}(x)=\phi(x)$. Se puede intentar resolver esta ecuación 'hacia atrás', y obtener al final la trayectoria óptima desde $x_0$, tomando como controles aquellos que minimizan la expresión anterior. A este enfoque se le llama programación dinámica. Este procedimiento resulta más eficiente que una búsqueda exhaustiva; sin embargo, la cantidad de operaciones escala a medida que aumenta la dimensionalidad del problema (es decir, este método posee la comunmente denominada 'maldición de dimensionalidad').

La programación dinámica diferencial (o DDP, por sus siglas en inglés) se presenta como un método iterativo que requiere solucionar 'de manera local' la ecuación de Bellman, realizando aproximaciones de segundo orden de $V$ y $S$, y obteniendo mínimos aproximados. En este sentido, este método se puede comparar con un método de tipo Newton. El método iLQR corresponde a eliminar ciertos términos de la expansión de $S$ que pueden resultar difíciles de calcular. En este sentido, se compara con el método de Newton-Raphson.

En el algoritmo original de (DDP), no se asume que se quiere llegar específicamente a algún punto $x^f$ en el tiempo final (o por lo menos en algunas de sus coordenadas). Lo anterior se puede intentar aproximar dando un costo final muy alto alrededor de los puntos $x_1^f,\ldots, x_q^f$. Aunque la introducción de funciones de este estilo generalmente conlleva una mayor inestibilidad en los cálculos numéricos (ver [1]), este enfoque se puede intentar aplicar para este algoritmo.

Todo el algoritmo que se presenta a continuación se basa en lo presentado dentro de la serie de videos "Optimal Control 2024", subidos por el laboratorio de exploración de robótica de la Universidad Carnegie Mellon, donde el profesor Zachary Manchester imparte clases sobre el tema. En particular, el siguiente código se construye a partir de lo presentado en la [sesión 10](https://www.youtube.com/watch?v=t0vaNTZIC20&list=PLZnJoM76RM6Jv4f7E7RnzW4rijTUTPI4u&index=10) y la [sesión 11](https://www.youtube.com/watch?v=qusvkcoHyz0&list=PLZnJoM76RM6Jv4f7E7RnzW4rijTUTPI4u&index=11) de esta serie.
"""

# ╔═╡ 6130a994-08c3-41f4-929c-ab9d92d72737
begin
	dimx=1
	dimu=1
	function f(x,u)
		return -x+u
	end
end

# ╔═╡ 0177b544-4500-4eed-9c0a-fe001e24705a
begin
	h=0.02;
	t_final=1;
	Nt=Int(div(t_final,h)+1);
	t_div=0:h:h*(Nt-1);
end

# ╔═╡ 43c83c86-f02a-48ae-9aa8-652ee4a3228c
###Aproximación mediante Runge-Kutta de roden 4 de la dinámica
function dynamics_rk4(x,u)
	k1=f(x,u)
	k2=f(x+0.5*h*k1,u)
	k3=f(x+0.5*h*k2,u)
	k4=f(x+h*k3,u)
	return x+h/6*(k1+2*k2+2*k3+k4)
end

# ╔═╡ 494776b0-ddae-4cf4-8e23-b1a888e9c716
###Derivadas de la dinámica
begin
	function dfdx(x,u)
		fx(z::Vector)=dynamics_rk4(z,u)
		ForwardDiff.jacobian(fx,x)
	end
	function dfdu(x,u)
		fu(w::Vector)=dynamics_rk4(x,w)
		ForwardDiff.jacobian(fu,u)
	end
	function dAdx(x,u)
		fAx(z::Vector)=vec(dfdx(z,u))
		ForwardDiff.jacobian(fAx,x)
	end
	function dAdu(x,u)
		fAu(w::Vector)=vec(dfdx(x,w))
		ForwardDiff.jacobian(fAu,u)
	end
	function dBdu(x,u)
		fBu(w::Vector)=vec(dfdu(x,w))
		ForwardDiff.jacobian(fBu,u)
	end
end

# ╔═╡ 559c1847-f136-4730-a0ba-9ec87e936b94
###Declaración de costos
begin
	Q=ones(dimx,dimx) ### Por si el costo es cuadrático en x
	R=ones(1,1) ###Por si el costo es cuadrático en u
	S=zeros(dimx,dimx) ##Se puede cambiar si se quiere alcanzar algún estado en especial
end

# ╔═╡ 785e4e5d-ad88-4721-bee0-1b6f694df719
function Comm(m, n)
    K = zeros(Int, m * n, m * n)
    for i in 1:m
        for j in 1:n
            k = (j - 1) * m + i
            l = (i - 1) * n + j
            K[k, l] = 1
        end
    end
    return K
end

# ╔═╡ 6b5c18d1-5963-4288-aaf0-3e769d1bb5b1
x_goal=[0] ##En este ejemplo, no hay estado final deseado.

# ╔═╡ 7bc7b335-fbbc-42ce-a349-ecf27ae9304f
#### Declaración del costo de etapas
function stage_cost(x,u)
	global x_goal
	return 0.5*(x-x_goal)'*Q*(x-x_goal)+0.5*u'*R*u
end

# ╔═╡ 018de8d3-12b5-4d31-9a4b-25eb2bf2a9e4
### Declaración del costo terminal
function terminal_cost(x)
	global x_goal
	return 0.5*(x-x_goal)'*S*(x-x_goal)
end

# ╔═╡ 7f02b86c-1328-4505-af1c-57c676a6b75f
function total_cost(x_traj,u_traj,init=1)
	L=0
	for k in init:Nt-1
		L = L+stage_cost(x_traj[:,k],u_traj[:,k])[1]
	end
	L+=terminal_cost(x_traj[:,Nt])[1]
	return L
end

# ╔═╡ 2ca34ace-4a13-475b-a90d-cadd4bf730c9
begin
	function backward_pass(x_traj,u_traj)
		d_loc=ones(dimx,Nt-1)
		K_loc=zeros(dimu,dimx,Nt)
		P_loc=zeros(dimx,dimx,Nt)
		p_loc=zeros(dimx,Nt)
		delta_J=0
		p_loc[:,Nt].=0 #S*(xtraj[:,Nt]-x_goal)
		P_loc[:,:,Nt]=S
		### Cálculo del backward-pass para k a partir del valor en k+1
		for k in Nt-1:-1:1
			l_x=Q*(x_traj[:,k]-x_goal) ##lx(z)=stage_cost(z,u[:,k])\\l_x=ForwardDiff.jacobian(lx,xtrahj[:,k])
			l_xx=Q
			##l_xx=ForwardDiff.hessian(lx,xtraj[:,k])
			l_xu=[0]
			##En caso de tener otra relación,implementar forwardDiff para hallar esta derivada.
			l_u=R*(u_traj[:,k]) #####lu(w)=stage_cost(x[:,k],w)\\l_u=ForwardDiff.jacobian(lx,utraj[:,k])
			l_uu=R
			##l_uu=ForwardDiff.hessian(lu,utraj[:,k])
			A=dfdx(x_traj[:,k],u_traj[:,k])
			B=dfdu(x_traj[:,k],u_traj[:,k])
	
			g_x=l_x+A'*p_loc[:,k+1]
			g_u=l_u+B'*p_loc[:,k+1]
			
			G_xx=l_xx+A'*P_loc[:,:,k+1]*A
			G_uu=l_uu+B'*P_loc[:,:,k+1]*B
			G_xu=l_xu+A'*P_loc[:,:,k+1]*B
	
			#COMENTAR SI SE HACE iLQR/ DESCOMENTAR PARA IMPLEMENTAR PROPIAMENTE DDF
			#Ax= dAdx(xtraj[:,k],utraj[:,k])
			#Au= dAdu(xtraj[:,k],utraj[:,k])
			#Bu= dBdu(xtraj[:,k],utraj[:,k])
			#G_xx+=kron(p_loc[:,k+1]',I(dimx))*Comm(dimx,dimu)*Ax
			#G_xu+=kron(p_loc[:,k+1]',I(dimx))*Comm(dimx,dimu)*Au
			#G_uu+=kron(p_loc[:,k+1]',I(dimx))*Comm(dimx,dimu)*Bu

			while !isposdef(G_uu)
				beta=0.0000001
				G_uu=G_uu+beta.*I(dimu,dimu)
				beta=2*beta
			end
			
			d_loc[:,k]=G_uu\g_u
			
			K_loc[:,:,k]=G_uu\G_xu
			P_loc[:,:,k]=G_xx+K_loc[:,:,k]'*G_uu*K_loc[:,:,k]-G_xu*K_loc[:,:,k]-K_loc[:,:,k]'*G_xu
			p_loc[:,k]=g_x-K_loc[:,:,k]'*g_u+K_loc[:,:,k]'*G_uu*d_loc[:,k]-G_xu*d_loc[:,k]
	
			delta_J+=dot(g_u,d_loc[:,k])
		end
		return d_loc,K_loc,p_loc,P_loc,delta_J
	end
end

# ╔═╡ 20842f67-8db2-4e64-85c4-f5b682c4c43e
begin
	global Nt,dimx,dimu
	function DDP_iLQR(xtraj, utraj, d=ones(Nt-1),K=zeros(dimu,dimx,Nt),P=zeros(dimx,dimx,Nt), p=zeros(dimx,Nt))
		xn=copy(xtraj)
		un=copy(utraj)
		d=ones(dimx,Nt-1)
		J=total_cost(xtraj,utraj)
		print("u inicial:",un,"\n\n")
		print("x inicial:",xn,"\n\n")
		print("Costo inicial:",J,"\n\n")
		iter=0
		m=0
		delta_costo=10
		while maximum(abs.(d[:]))>1.e-3
			m+=1
			iter+=1
			d,K,p,P,delta_J=backward_pass(xtraj,utraj)
			print("d_",iter,":",d, "\n")
			xn[:,1].=xtraj[:,1]
			alpha=1
			### Se realiza la dinámica con el nuevo valor del control 
			for k in 1:(Nt-1)
				un[:,k]=utraj[:,k]-alpha.*d[:,k]-K[:,:,k]*(xn[:,k]-xtraj[:,k])
				xn[:,k+1]=dynamics_rk4(xn[:,k],un[:,k])
			end
			Jn=total_cost(xn,un)
			println("Costo:",Jn,"\n")
			###Se evalua si se mejora; en caso, contrario se realiza búsqueda lineal sobre alpha repitiendo el algoritmo.
			while isnan(Jn[1]) || Jn>(J-1e-4*alpha*delta_J)
				alpha=0.2*alpha
				### Se realiza la dinámica con el nuevo valor del control 
				for k in 1:(Nt-1)
					un[:,k]=utraj[:,k]-alpha.*d[:,k]-K[:,:,k]*(xn[:,k]-xtraj[:,k])
					xn[:,k+1]=dynamics_rk4(xn[:,k],un[:,k])
				end
				Jn=total_cost(xn,un)
			end
			delta_costo=abs(J-Jn)
			J=Jn
			xtraj=copy(xn)
			utraj=copy(un)
			if m%5==0
				norma_infinito = maximum(abs.(d[:]))
				println("Valor de ||d||∞ en la iteración ", m, ": ", norma_infinito)
				println("Costo:",J)
			end
			if iter>1000
				print("Se excedió el número de iteraciones permitidas. El método no ha convergido")
				break
			end
		end
		println("Proceso completado.\nIteraciones realizadas:",iter, "\n\n")
		println("x óptimo:",xtraj,"\n")
		println("u óptimo",utraj)
		
		return(xtraj,utraj,d,J,p,P)
	end
end

# ╔═╡ 93e5bc1f-8874-46fc-85c6-9f963de80008
begin
####Definimos parametros iniciales
	x0=[1]
	xtraj=zeros(dimx, Nt)
	xtraj[:,1]=x0
	utraj=zeros(dimu, Nt-1)
	for k in 2:Nt
		xtraj[:,k]=dynamics_rk4(xtraj[:,k-1],utraj[:,k-1])
	end
end

# ╔═╡ 830738da-3a54-454b-93a6-472b07a060e0
xopt,uopt, dopt, Jopt, popt,Popt= DDP_iLQR(xtraj, utraj)

# ╔═╡ 6ebb5464-8c8a-4bbf-bc85-b71fbb8f186a
xtraj

# ╔═╡ 305f3e99-2682-4a50-aaf8-d35b5e2ef834
function sol(t)
    numerador = sqrt(2) * cosh(sqrt(2) * (t - 1)) - sinh(sqrt(2) * (t - 1))
    denominador = sqrt(2) * cosh(sqrt(2)) + sinh(sqrt(2))
    return numerador / denominador
end

# ╔═╡ 40a71cb0-c35f-4223-9483-075676f7d6d1
function control_real(t)
  return sinh(sqrt(2)*(t-1))/(sqrt(2)*cosh(sqrt(2)) + sinh(sqrt(2)))
end

# ╔═╡ 2acb8668-4672-449a-a1ea-ff7f20b1246b
begin
	Plots.plot(t_div, sol.(t_div),label="Solución real",lw=3, ls=:dash)
	Plots.plot!(t_div,xopt[1,:],title="Comparación del estado obtenido y\n el estado óptimo real", label="Solución obtenida" ,lw=2)
end

# ╔═╡ ef0f10cc-13f5-4c4b-a21f-928c04761cd1
begin
	Plots.plot(t_div[1:Nt-1], control_real.(t_div[1:Nt-1]),label="Control real",lw=3, ls=:dash)
	Plots.plot!(t_div[1:Nt-1],uopt[1,:],title="Comparación del control obtenido y\n el control óptimo", label="Control obtenido mediante DDP" ,lw=1)
end

# ╔═╡ 0fb079c3-3e4f-4036-a632-ad439313711a
begin
    x_vf = range(0, 1.5, length=50) 
    function V(x, k)
        Pk = Popt[:, :, k]   
        pk = popt[:, k]      
        return (0.5 * (x .-xopt[:,k])' * Pk * (x .-xopt[:,k]) + pk' * (x .-xopt[:,k]) .+ total_cost(xopt, uopt, k))[1]
    end
    Z = [V(x_i, k) for k in 1:Nt, x_i in x_vf]
    nothing  
end

# ╔═╡ 81413454-4e44-4467-9a0f-b0b4a5207b06
begin
####Definimos parametros iniciales
	x01=[1.5]
	x2traj=zeros(dimx, Nt)
	x2traj[:,1]=x01
	u2traj=ones(dimu, Nt-1)
	for k in 2:Nt
		x2traj[:,k]=dynamics_rk4(x2traj[:,k-1],u2traj[:,k-1])
	end
end

# ╔═╡ 85ba188c-f69f-485a-8b86-4b60b07e4149
x2opt,u2opt, d2opt, J2opt, p2opt,P2opt= DDP_iLQR(x2traj, u2traj)

# ╔═╡ 7aa982db-0ca8-41ce-850d-e6d916a5d73c
begin
####Definimos parametros iniciales
	x02=[2]
	x3traj=zeros(dimx, Nt)
	x3traj[:,1]=x02
	u3traj=ones(dimu, Nt-1)
	for k in 2:Nt
		x3traj[:,k]=dynamics_rk4(x3traj[:,k-1],u3traj[:,k-1])
	end
end

# ╔═╡ 7689b3ec-6bdf-4d61-9317-29af64eb71db
x3opt,u3opt, d3opt, J3opt, p3opt,P3opt= DDP_iLQR(x3traj, u3traj)

# ╔═╡ f70eb405-2f45-4b28-95b3-ea78b02f9b34
begin
	Plots.plot(t_div,xopt[1,:],title="Trayectoria que soluciona el Principio de Pontryagin\ncon x(0)=[1]", label="" ,lw=2, color=:red, xlabel="t", ylabel="x")
end

# ╔═╡ 5aef0348-61c6-4fd6-bfbc-c4356dcda30b
begin
	Plots.plot(t_div,xopt[1,:],title="Algunas curvas de una familia que cubre un dominio [0,1]xR" ,lw=2,label="Trayectoria óptima para x(0)=$x0",color=:red, xlabel="t", ylabel="x")
	Plots.plot!(t_div,x2opt[1,:],lw=2,label="Trayectoria óptima para x(0)=$x01",color=:blue)
	Plots.plot!(t_div,x3opt[1,:] ,lw=2,label="Trayectoria óptima para x(0)=$x02",color=:orange)
end

# ╔═╡ ffd94de9-7f13-4b5c-98b5-c232d9571b32
md"""
Adicionalmente, este método nos permite aproximar los valores de $V(t,x)$ alrededor de la trayectoria óptima por medio de una expansión de Taylor de orden 2. Esto es útil, por ejemplo, para cuando en la vida real se aplica un control que, en teoría, es óptimo, pero resulta, por motivos de aleatoriedad, en un estado distinto, pero cercano, al esperado. $V(t,x)$ nos permite estimar el mínimo costo que esperamos asumir con esta variación.
"""

# ╔═╡ abae767c-d3e1-46b5-98e4-9737df833088
begin
	plt=Plots.surface(t_div,x_vf, Z', xlabel="t", ylabel="x", zlabel="V(t,x)", title="V(t,x) formado a partir de trayectorias óptimas",legend = :topright)
	V_cero = [V(xopt[1,k], k) for k in 1:Nt] 
	Plots.plot!(plt,  t_div, xopt[1,:], V_cero, label="Trayectoria óptima para x(0)=$x0",linewidth=1, color=:red)
	Plots.plot!(plt,  t_div, x2opt[1,:], V_cero, label="Trayectoria óptima para x(0)=$x01",linewidth=1, color=:blue)
	Plots.plot!(plt,  t_div, x3opt[1,:], V_cero, label="Trayectoria óptima para x(0)=$x02",linewidth=1, color=:orange)
end

# ╔═╡ 27e9c6f5-0abf-4f54-aafa-524dfb9b9037
begin
    plt_1 = Plots.plot3d(xlabel="t", ylabel="x", zlabel="V(t,x)", 
                       title="Trayectoria óptima", legend=:topright)

    # Graficar la trayectoria óptima en 3D
    Plots.plot3d!(plt_1, t_div, xopt[1, :], V_cero, label="Evaluación sobre trayectoria óptima x(0)=$x0", linewidth=2, color=:red)

    # Proyección en el plano (t, x) -> línea amarilla en z=0
    Plots.plot3d!(plt_1, t_div, xopt[1, :], fill(0, length(t_div)), 
                  label="", linewidth=2, color=:red, linestyle=:dash)

    # Líneas verticales desde la proyección a la trayectoria
    for k in 1:Nt
        Plots.plot3d!(plt_1, [t_div[k], t_div[k]], [xopt[1, k], xopt[1, k]], [0, V_cero[k]], 
                      color=:red, linewidth=0.3, label="")
    end



	
    Plots.plot3d!(plt_1, t_div, x2opt[1, :], V_cero, label="Evaluación sobre trayectoria óptima para x(0)=$x01", linewidth=2, color=:blue)

    # Proyección en el plano (t, x) -> línea amarilla en z=0
    Plots.plot3d!(plt_1, t_div, x2opt[1, :], fill(0, length(t_div)), 
                  label="", linewidth=2, color=:blue, linestyle=:dash)

    # Líneas verticales desde la proyección a la trayectoria
    for k in 1:Nt
        Plots.plot3d!(plt_1, [t_div[k], t_div[k]], [x2opt[1, k], x2opt[1, k]], [0, V_cero[k]], 
                      color=:blue, linewidth=0.3, label="")
    end
	for k in 1:Nt
        Plots.plot3d!(plt_1, [t_div[k], t_div[k]], [xopt[1, k], xopt[1, k]], [0, V_cero[k]], 
                      color=:blue, linewidth=0.3, label="")
    end


	
    Plots.plot3d!(plt_1, t_div, x3opt[1, :], V_cero, label="Evaluación sobre trayectoria óptima para x(0)=$x02", linewidth=2, color=:orange)

    # Proyección en el plano (t, x) -> línea amarilla en z=0
    Plots.plot3d!(plt_1, t_div, x3opt[1, :], fill(0, length(t_div)), 
                  label="", linewidth=2, color=:orange, linestyle=:dash)

    # Líneas verticales desde la proyección a la trayectoria
    for k in 1:Nt
        Plots.plot3d!(plt_1, [t_div[k], t_div[k]], [x3opt[1, k], x3opt[1, k]], [0, V_cero[k]], 
                      color=:orange, linewidth=0.3, label="")
    end
	Plots.plot3d!()
end

# ╔═╡ 265ecaf9-f2b4-42a3-9766-6db564c951d1
md"""
**Gráfico interactivo:** Si ejecuta este código, puede usar el mouse para mover el siguiente gráfico con el botón derecho, izquierdo, y rueda central. Presionando el botón z en su teclado, puede rotar el gráfico. 
"""

# ╔═╡ b0f3e68a-059a-4b1a-983a-318317372238
begin
	fig = WGLMakie.Figure()
	ax = WGLMakie.Axis3(fig[1,1], 
	                    xlabel="t", ylabel="x", zlabel="V(x,t)", 
	                    title="V(t,x) alrededor de la trayectoria óptima",titlesize=20)
	
	WGLMakie.surface!(ax, t_div, x_vf, Z)
	
	WGLMakie.lines!(ax, t_div, xopt[1, :], V_cero, 
	                linewidth=2, color=:red, label="Valor sobre la \ntrayectoria óptima")
	
	leg = Legend(fig[1, 2], ax, 
		labelsize=10,
            width = 100, height = 50,
            halign = :left, valign = :top, framevisible = false)                   
	fig
end

# ╔═╡ 95aa2db6-1f1b-49e6-97de-700912d87b31
md"""
# Algoritmo de búsqueda del vecino más extremo con uso de matriz de transición
"""

# ╔═╡ ced75807-dec6-4d39-9a5a-690238c29272
md"""
Ahora, se intentará solucionar el sistema de ecuaciones diferenciales con doble frontera dado por el Principio de Pontryaguin sin restricciones en el control. Recordemos que definimos el *hamiltoniano*

$H(x,\lambda,u):=L(t,x,u)+\lambda f(t,x,u).$
Suponga que, además de las definiciones dadas más arriba, queremos llegar a algunos estados finales, que expresamos para $q\leq n$ mediante la función

$\psi(x(t_f))=\begin{pmatrix}x_1(t_f)-x_1'\\ 
\vdots \\
x_{q}(t_f)-x^f_q \end{pmatrix}=0, \quad x_1^f,\ldots, x_q^f \text{ fijos.}$

El Principio de Pontryagin enuncia que existe una función $\lambda: [0,t_f]\to \mathbb{R}^n$ de manera que la trayectoria óptima satisface el sistema de ecuaciones diferenciales con doble frontera (para $t_0$,$t_f$ fijos)

$\begin{cases}
\dot{x}=\frac{\partial H}{\partial \lambda}=f(t,x,u),\\
\dot{\lambda}=-\frac{\partial H}{\partial x}=-\left(\frac{\partial f}{\partial x}\right)^T\lambda-\left(\frac{\partial L}{\partial x}\right),\end{cases}$
donde

$H(t,x,u)=\min_{v\in U}\left[H(t,x,v)\right],$
y además se cumplen las condiciones (de doble frontera)

$\begin{align}\lambda_j(t_f)&=\left(\frac{\partial \phi}{\partial x_j}\right)_{t=t_f},\quad j=q+1,\ldots, n.\\
x(t_0)&=x_0.\end{align}$

Note que, si no se tiene restricciones sobre el control y el hamilitoniano no es lineal respecto a $u$, entonces la condición sobre el mínimo es equivalente a

$0=\left(\frac{\partial f}{\partial u}\right)^T\lambda+\left(\frac{\partial L}{\partial u}\right).$

Bajo estas condiciones, note que quedan $n$ condiciones iniciales indeterminadas $\lambda_1(t_0),\ldots, \lambda_n(t_0),$ y $n$ condiciones terminales indeterminadas $\lambda_1(t_f), \ldots, \lambda_{q}(t_f), x_{q+1}(t_f), \ldots, x_n(t_f).$ 

El algoritmo que se presenta a continuación plantea la opción de estudiar la función $\Lambda:\mathbb{R}^n\to \mathbb{R}^n$ que envia estas condiciones iniciales en su respectivo error respecto a las condiciones finales. Es decir, dadas unas condiciones iniciales $\lambda_0:=[\lambda_n(t_0),\ldots, \lambda_n(t_0)]^T$, si $\xi:[0,t_f]\to \mathbb{R}^n$ y $\mu:[0,t_f]\to \mathbb{R}^n$ son las funciones que se generan al integrar la dinámica dada por el principio de Pontryagin con condiciones iniciales $\xi(0)=x_0$ y $\mu(0)=\lambda_0$, queremos estudiar los ceros función $\Lambda: \mathbb{R}^n\to \mathbb{R}^n$ dada por

$\lambda_0\to \Lambda(\lambda_0)=\begin{pmatrix}\xi_1(t_f)-x_1^f\\
\vdots \\
\xi_1(t_f)-x_1^f\\
\mu_{q+1}(t_f)-(\partial\phi/\partial x_{q+1})_{t=t_f}\\
\vdots\\
\mu_{n}(t_f)-(\partial\phi/\partial x_n)_{t=t_f}\end{pmatrix}.$

Para este último propósito existen varios métodos (prácticamente, tantos como existan de hallar raíces de una función de "caja negra"). Una aproximación puede bisección, secante o regula falsi (para resultados con esta metodología, ver [4]). En este cuaderno, se implementará algo más similar a una iteración tipo Newton, en donde, después de calcular una estimación inicial, se estima la diferencial de esta función alrededor de cada punto, y se avanza en la dirección que, según la linearización, reduce el tamaño de $\Lambda$. Nótese que si en efecto los extremos encontrados son óptimos, entonces este método calcula por medio de una clase de 'método' de las características, algunos valores de la función de valor próximo (*cost-to-go*) sobre la vecindad de la variedad diferenciable de llegada de las trayectorias.
"""

# ╔═╡ 35c66f76-7100-40d9-85f8-cfac9796b169
md"""
En nuestro ejemplo en específico, note que la ecuación adjunta está dada por 

$\dot{\lambda}=-x+\lambda, \textup{ y }0=u+\lambda,$

donde queremos llegar a la condición final $\lambda(1)=0$.
"""

# ╔═╡ c9846053-4131-4df7-b855-25633efaa0d1
begin
	####Función de la dinámica adjunta
	function adj_f(lambda,x,u)
		return -x+lambda
	end
	###Aproximación mediante Runge-Kutta de roden 4 de la dinámica
	function adj_dynamics_rk4(lambda,x,u)
		k1=adj_f(lambda,x,u)
		k2=adj_f(lambda+0.5*h*k1,x,u)
		k3=adj_f(lambda+0.5*h*k2,x,u)
		k4=adj_f(lambda+h*k3,x,u)
		return lambda+h/6*(k1+2*k2+2*k3+k4)
	end
	###Función que simula la dinámica completa tomando un punto inicial lambda_0
	function total_dynamics(lambda0::Vector)
		global dimx,dimu,Nt
		###Declaración de las variables
		x_traj = zeros(eltype(lambda0), dimx, Nt)
    	lamb_traj = zeros(eltype(lambda0), dimx, Nt)
    	u_traj = zeros(eltype(lambda0), dimu, Nt-1)
		lamb_traj[:,1]=lambda0
		x_traj[:,1]=x0
		#####Se toma u minimizando el hamiltoniano (cambiar según la dificultad del problema. Puede requerir algoritmos de optimización).
		u_traj[:,1]=-lamb_traj[:,1]
		for k in 2:Nt
			x_traj[:,k]=dynamics_rk4(x_traj[:,k-1],u_traj[:,k-1])
			lamb_traj[:,k]=adj_dynamics_rk4(lamb_traj[:,k-1], x_traj[:,k-1], u_traj[:,k-1])
			#####Se toma u minimizando el hamiltoniano (cambiar según la dificultad del problema. Puede requerir algoritmos de optimización).
			if k<Nt
				u_traj[:,k]=-lamb_traj[:,k]
			end
		end
		return u_traj,x_traj,lamb_traj
	end
		
	###Función que lleva condiciones iniciales en las condiciones finales usando la dinámica de ambas condiciones.
	function Lambda(lambda0::Vector)
		u_obt,x_obt,lamb_obt=total_dynamics(lambda0)
		####Al final, se retorna el vector de comparación entre los estados finales a los que se llega en comparación a los deseados. En nuestro caso, se requiere únicamente que lambda_term=0. En otros casos, se puede modificar esta parte del código según se crea conveniente.
		return lamb_obt[:,Nt]
	end
end

# ╔═╡ 22bff809-b3fd-499f-82ba-04341f72b4b5
md"""
A continuación, haciendo uso del método de diferencias hacia adelante (forward differences), se realiza una función que calcula el jacobiano de nuestra función Lambda en un punto $\lambda_0$. Dependiendo del valor de $\lambda$, esta matriz puede llegar a ser mal condicionada (ver [1]). Esto, sumado a que las dinamicas principal y adjunta pueden diferir de manera extrema en ordenes dependiendo del estado incial de la variable adjunta, es uno de los principales problemas de este método. Por ello, la estabilidad depende de la elección inicial de $\lambda_0$. 
"""

# ╔═╡ f3935d3a-07fd-4036-8e66-a7a61936df61
function difLambda(lambda0::Vector)
	return ForwardDiff.jacobian(Lambda,lambda0)
end

# ╔═╡ d8fae788-1ab7-4a67-8bc0-227f24311aa2
function neigh_extremals_newton(lambda0)
	tol=1
	new_lambda=lambda0
	it=0
	while tol>1.e-4
		it+=1
		imag=Lambda(new_lambda)
		tol=maximum(abs.(imag))
		grad=difLambda(new_lambda)
		if cond(grad)>1.e6
			beta=0.01
			grad=grad+beta.*I(dimx,dimx)
			beta=2*beta
		end
		alpha=1
		dir_des=- grad\imag
		delta_lambda=alpha .*dir_des
		lambdan=copy(new_lambda+delta_lambda)
		while maximum(abs.(Lambda(lambdan)))>tol
			alpha=0.5*alpha
			delta_lambda=alpha.* dir_des
			lambdan=copy(new_lambda+delta_lambda)
		end
		new_lambda=copy(lambdan)
	end
	u_obt, x_obt,lamb_obt=total_dynamics(new_lambda)
	print("Número de iteraciones requeridas: ",it)
	return u_obt,x_obt,lamb_obt, new_lambda
end

# ╔═╡ 3c8814e3-2ebf-4c4d-b136-1f8827ef0e5d
u_opt_2,x_opt_2,lambd_opt_2, lambda0=neigh_extremals_newton([-0.3])

# ╔═╡ 3f150fc1-93db-4602-ac10-d42b2ae754af
begin
	Plots.plot(t_div, sol.(t_div),label="Solución real",lw=3, ls=:dash)
	Plots.plot!(t_div,x_opt_2[1,:],title="Comparación de la solución obtenida y\n la solución real", label="Solución obtenida" ,lw=2)
end

# ╔═╡ 1e464469-e181-4b6d-a986-4566dfdfcd0b
begin
	Plots.plot(t_div, -control_real.(t_div),label="Solución adjunta real",lw=3, ls=:dash)
	Plots.plot!(t_div,lambd_opt_2[1,:],title="Comparación de la dinámica adjunta y\n la solución adjunta real", label="Solución adjunta obtenida" ,lw=2)
end

# ╔═╡ c686379e-22e1-49dc-95a7-14f14113df7d
begin
	Plots.plot(t_div[1:Nt-1], control_real.(t_div[1:Nt-1]),label="Control óptimo real", ls=:dash, lw=3)
	Plots.plot!(t_div[1:Nt-1],u_opt_2[1,:],title="Comparación del control obtenido mediante NE\n y control óptimo real", label="Control obtenido con NE",lw=2)
end

# ╔═╡ 32bf6afa-5af1-4372-9796-1ddb251ea119
md"""
# Búsqueda del vecino más extremo a partir de valores finales
"""

# ╔═╡ e4d073b8-4335-4417-ba5e-e751b000bb20
md"""
En [1], se propone un método numérico similar al anterior, pero que no requiere el cálculo numérico de derivadas, y por tanto no depende de la estabilidad numérica de las derivadas. En este libro, se muestra que se puede aproximar el cambio diferencial  que genera una modificación en las condiciones finales $\delta x(t_f)$ y $\delta \lambda(t_f)$ de las ecuaciones ecuaciones anteriores sobre las condiciones iniciales $x(t_0)$ y $\lambda(t_0)$. El algoritmo sigue de la siguiente manera.

a) Estime los $q$ parámetros $\nu^T=[\lambda_1(t_f),\ldots, \lambda_q(t_f)]$, y los $n-q$ valores no especificados en este tiempo final de la variable de estado $[x_{q+1}(t_f),\ldots, x_n(t_f)]$.

b) Integre la dinámica de estado y la dinámica adjunta **hacia atrás** con las condiciones en $t_f$ dadas junto con las estimadas inicialmente.

c) Simultaneamente, resuelva las ecuaciones diferenciales matriciales

$\begin{align}\dot{S}&=-SA+A^TS+SBS-C\\
\dot{R}&=-(A^T-SB)R\\
\dot{Q}&=R^TBR,
\end{align}$
donde las condiciones finales están dadas por $S(t_f)=\phi_{xx}\big|_{t=t_f}$, $R(t_f)=Id$ y $Q(t_f)=0$, y 

$\begin{align}A(t)=f_x-f_uH_{uu}^{-1}H_{ux}\\
B(t)=f_uH_{uu}^{-1}f_u^T, \text{ y }\\
C(t)=H_{xx}-H_{xu}H_{uu}^{-1}H_{ux}.\end{align}$

d) Guarde localmente $x(t_0), \lambda(t_0)$ y $(S-R^TQ^{-1}R)\big|_{t=t_0}$. Escoja $\delta x(t_0)$ de tal manera que se acerque más a los valores deseados de $x_0$. En [1], se demuestra que $\delta \lambda(t_0)=(S-R^TQ^{-1}R)\big|_{t=t_0}\delta x(t_0).$

e) En el mismo texto, se demuestra que

$\begin{align}
\delta \dot{x}=A(t)\delta x-B(t)\delta\lambda\\
\delta\dot{\lambda}=-C(t)\delta x-A(t)^T\delta\lambda.\end{align}$

Con las condiciones iniciales elegidas en el anterior literal, puede resolver este sistema y obtener los valores de $\nu^T:=[\delta \lambda_1(t_f)\ldots \lambda_q(t_f)]$ y $[\delta x_{q+1}(t_f)\ldots, \delta x_{n}(t_f)].$

g) Actualice los valores estimados en el paso (a) tomando $\nu_{\text{new}}=\nu_{\text{old}}+dv$ y $(x_i(t_f))_{\text{new}}=(x_i(t_f))_{\text{old}}+\delta x_i(t_f)$ para $i=q+1,\ldots,n$. Repita los pasos hasta obtener una precisión deseada. 
"""

# ╔═╡ ca26ab49-1c23-4f6f-b2c3-fb84fce0870e
md"""
En nuestro ejemplo, dado que trabajamos en el caso unidimensional, denotaremos las matrices opor sus respectivas letras en minúscula. Para este caso, $a(t)=-1$, y $b(t)=c(t)=1$, y debemos resolver

$\begin{align}
\dot{s}=s^2-1,\\
\dot{r}=(1+s)r,\\
\dot{q}=r^2,
\end{align}$ 

con $s(1)=0$, $r(1)=1$ y $q(1)=0$.
"""

# ╔═╡ 4604cf4b-0750-4f0d-80b4-5781df324621
####DEFINICIÓN DE LAS DINÁMICAS
begin
	function reverse_dynamics_rk4(x,u)
		k1=f(x,u)
		k2=f(x-0.5*h*k1,u)
		k3=f(x-0.5*h*k2,u)
		k4=f(x-h*k3,u)
		return x-h/6*(k1+2*k2+2*k3+k4)
	end
	function adj_rev_dynamics_rk4(lambda,x,u)
		k1=adj_f(lambda,x,u)
		k2=adj_f(lambda-0.5*h*k1,x,u)
		k3=adj_f(lambda-0.5*h*k2,x,u)
		k4=adj_f(lambda-h*k3,x,u)
		return lambda-h/6*(k1+2*k2+2*k3+k4)
	end
	function f_s(s,r,q)
		return s^2-1
	end
	function f_r(s,r,q)
		return (1+s)*r
	end
	function f_q(s,r,q)
		return r^2
	end
	s=zeros(1,Nt)
	r=zeros(1,Nt)
	q=zeros(1,Nt)
	s[:,Nt]=[0]
	r[:,Nt]=[1]
	q[:,Nt]=[0]
	function rk4_rev_matrix(s,r,q)
		s_1=f_s(s,r,q)
		r_1=f_r(s,r,q)
		q_1=f_q(s,r,q)
		s_2=f_s(s-0.5*h*s_1,r-0.5*h*r_1, q-0.5*h*q_1)
		r_2=f_r(s-0.5*h*s_1,r-0.5*h*r_1, q-0.5*h*q_1)
		q_2=f_q(s-0.5*h*s_1,r-0.5*h*r_1, q-0.5*h*q_1)
		s_3=f_s(s-0.5*h*s_2,r-0.5*h*r_2, q-0.5*h*q_2)
		r_3=f_r(s-0.5*h*s_2,r-0.5*h*r_2, q-0.5*h*q_2)
		q_3=f_q(s-0.5*h*s-2,r-0.5*h*r_2, q-0.5*h*q_2)
		s_4=f_s(s-h*s_3,r-h*r_3, q-h*q_3)
		r_4=f_r(s-h*s_3,r-h*r_3, q-h*q_3)
		q_4=f_q(s-h*s_3,r-h*r_3, q-h*q_3)
		return s-h/6*(s_1+2*s_2+2*s_3+s_4), r-h/6*(r_1+2*r_2+2*r_3+r_4), q-h/6*(q_1+2*q_2+2*q_3+q_4)
	end
	for k in Nt-1:-1:1
		s[1,k],r[1,k],q[1,k]=rk4_rev_matrix(s[1,k+1],r[1,k+1],q[1,k+1])
	end

	function total_reverse_dynamics(xfinal::Vector)
		global dimx,dimu,Nt
		###Declaración de las variables
		x_traj = zeros(dimx, Nt)
    	lamb_traj = zeros(dimx, Nt)
    	u_traj = zeros(dimu, Nt)
		x_traj[:,Nt]=xfinal
		###lambdatraj[j,1]=(\partial \phi/\partial x_j)|(t=t_f)
		lamb_traj[:,1]=[0]
		#####Se toma u minimizando el hamiltoniano (cambiar según la dificultad del problema. Puede requerir algoritmos de optimización).
		u_traj[:,Nt]=-lamb_traj[:,Nt]
		for k in Nt-1:-1:1
			x_traj[:,k]=reverse_dynamics_rk4(x_traj[:,k+1],u_traj[:,k+1])
			lamb_traj[:,k]=adj_rev_dynamics_rk4(lamb_traj[:,k+1], x_traj[:,k+1], u_traj[:,k+1])
			#####Se toma u minimizando el hamiltoniano (cambiar según la dificultad del problema. Puede requerir algoritmos de optimización).
			u_traj[:,k]=-lamb_traj[:,k]
		end
		return u_traj,x_traj,lamb_traj
	end	
	function integrate_forward_deltas(delta_x0,delta_lambda0)
		dxtraj=zeros(dimx, Nt)
		dlamb_traj=zeros(dimx, Nt)
		dxtraj[:,1]=delta_x0
		dlamb_traj[:,1]=delta_lambda0
		function f_dx(x,lambda)
			return -x-lambda
		end
		function f_dlambda(x,lambda)
			return -x+lambda
		end
		function rk4deltas(x,lambda)
			k1=f_dx(x,lambda)
			l1=f_dlambda(x,lambda)
			k2=f_dx(x+0.5*h*k1,lambda+0.5*h*l1)
			l2=f_dlambda(x+0.5*h*k1,lambda+0.5*h*l1)
			k3=f_dx(x+0.5*h*k2,lambda+0.5*h*l2)
			l3=f_dlambda(x+0.5*h*k2,lambda+0.5*h*l2)
			k4=f_dx(x+h*k3,lambda+h*l3)
			l4=f_dlambda(x+h*k3,lambda+h*l3)
			return x+h/6*(k1+2*k2+2*k3+k4), lambda+h/6*(l1+2*l2+2*l3+l4)
		end
		for k in 1:Nt-1
			dxtraj[:,k+1], dlamb_traj[:,k+1]=rk4deltas(dxtraj[:,k], dlamb_traj[:,k])
		end
		return dxtraj[:,Nt], dlamb_traj[:,Nt]
	end
end

# ╔═╡ 37d29234-93e6-4700-9ab5-493cbf929d06
Plots.plot(t_div,[s[1,:],r[1,:],q[1,:]], title="Soluciones de la ecuación diferencial matricial\nasociada al problema", label=["s" "r" "q"])

# ╔═╡ 5ca07f66-3099-4b9e-8f2f-dd857fbb2d08
begin
	s_imp=s[1,1]
	r_imp=r[1,1]
	q_imp=q[1,1]
	### VECINO MÁS EXTREMO CON VALORES FINALES
	function vecino_mas_extremo_cf(xfinal)
		tol=1
		xn=copy(xfinal)
		u_traj=zeros(1,Nt)
		x_traj=zeros(1,Nt)
		l_traj=zeros(1,Nt)
		it=0
		while maximum(abs.(tol))>1.e-6
			it+=1
			u_tra,x_tra,l_tra=total_reverse_dynamics(xn)
			x0_obt=x_tra[:,1]
			tol=x0_obt-x0
			delta_lambda0=(s[:,1]-r[:,1]'*(q[:,1]\r[:,1]))*tol
			deltaxf, deltalambf=integrate_forward_deltas(tol,delta_lambda0)
			xn+=deltaxf
			u_traj=copy(u_tra)
			x_traj=copy(x_tra)
			l_traj=copy(l_tra)
		end
		print("Número de iteraciones requeridas: ",it)
		return u_traj, x_traj, l_traj
	end
end

# ╔═╡ 166ed9f2-2cc5-493a-8160-76c7977e8006
uopt3, xopt3,lopt3=vecino_mas_extremo_cf([0.2])

# ╔═╡ 3fbcea27-1585-4bce-abde-f37b8c382ced
begin
	Plots.plot(t_div, sol.(t_div),label="Solución real",lw=3, ls=:dash)
	Plots.plot!(t_div,xopt3[1,:],title="Comparación de la solución obtenida con NE estable y\n la solución real", label="Solución obtenida" ,lw=2)
end

# ╔═╡ e8565754-334e-4447-b064-bb340cb6f8d9
begin
	Plots.plot(t_div, -control_real.(t_div),label="Solución adjunta real",lw=3, ls=:dash)
	Plots.plot!(t_div,lopt3[1,:],title="Comparación de la dinámica adjunta obtenida con\n NE estable y la solución adjunta real", label="Solución adjunta obtenida" ,lw=2)
end

# ╔═╡ 442b2a19-9a17-44b7-b4df-91062b6bb435
begin
	Plots.plot(t_div[1:Nt], control_real.(t_div[1:Nt]),label="Control óptimo real", ls=:dash, lw=3)
	Plots.plot!(t_div[1:Nt],uopt3[1,:],title="Comparación del control obtenido mediante\n NE estable y control óptimo real", label="Control obtenido con NE",lw=2)
end

# ╔═╡ 9ca1805e-c474-4139-bd7f-6ed8f44b647b
md"""
# Método de barrido hacia adelante y hacia atrás (FBS)
"""

# ╔═╡ 87312b77-b624-47f4-9a2b-790d73cde8f6
md"""
La inspiración para la implementación de este método se toma de [4]. Este se caracteriza por trabajar sobre los controles de una trayectoria, e iterarlos basándose en el principio de Pontryagin. En principio, como en DDP, asume que no se requiere llegar exactamente a un estado final $x_f$. Recuerde que, como ya se ha discutido, esto se puede intentar lograr con una función terminal $\phi$ que asigne costos altos a puntos diferentes al deseado. No obstante, la introducción de este tipo de funciones conlleva una mayor inestabilidad en los cálculos numéricos.

Se inicia con una estimación inicial del control $u_0$. Dada una estimación del control $u_n$, se estima las trayectoria principal $x_n$ a partir de la dinámica $x(0)=x_0$ y $\dot{x}=f(t,x,u)$ (barrido "*hacia adelante*"). Posteriormente, con estos valores, se estima la trayectoria adjunta $\lambda_n$ con la dinámica $\lambda(t_f)=\partial\phi/\partial x\big|_{t=t_f}$ y $\dot{\lambda}=-\partial H/\partial x$ (barrido "*hacia atrás*"). Posteriormente, actualizamos los valores del control según la propiedad minimizante. Para ello, definimos $u_{n}^{new}$ como

$u_{n}^{new}=\min_{v\in U}[H(t,x_n,\lambda_n,v)],$
y realizamos así la actualización

$u_{n+1}=\epsilon u_n+(1-\epsilon)u_n^{new},$
donde $0\leq\epsilon<1.$
Estos pasos se repiten hasta que se logre una tolerancia deseada (la diferencia entre los controles obtenidos sea suficientemente pequeña).
"""

# ╔═╡ 52e99cba-6a2a-43ea-ae34-c6d15ed38683
function forward_backward_sweep(u0)
	tol=1
	un=copy(u0)
	epsilon=1e-2
	xn=zeros(dimx,Nt)
	lambdan=zeros(dimx,Nt)
	iter=0
	while maximum(abs.(tol))>1e-6
		iter+=1
		xn[:,1]=x0	
		#Barrido 'hacia adelante'
		for k in 1:Nt-1
			xn[:,k+1]=dynamics_rk4(xn[:,k],un[:,k])
		end
		lambdan[:,Nt]=[0]
		#Barrido 'hacia atrás'
		for k in Nt-1:-1:1
			lambdan[:,k]=adj_rev_dynamics_rk4(lambdan[:,k+1], xn[:,k+1], un[:,k+1])
		end
		u_new=-lambdan ##propiedad minimizante (cambiar según el problema)
		tol=(1-epsilon).* (u_new - un)
		un=(1-epsilon) .* u_new+epsilon .*un
	end
	for k in 1:Nt-1
		xn[:,k+1]=dynamics_rk4(xn[:,k],un[:,k])
	end
	lambdan[:,Nt]=[0]
	#Barrido 'hacia atrás'
	for k in Nt-1:-1:1
		lambdan[:,k]=adj_rev_dynamics_rk4(lambdan[:,k+1], xn[:,k+1], un[:,k+1])
	end
	print("Número total de iteraciones: ",iter)
	return xn, lambdan, un
end

# ╔═╡ 6d933be0-8221-49c7-8776-3d5d27d0d943
begin
	u0=ones(1,Nt)
	xopt4, lambopt4, uopt4= forward_backward_sweep(u0)
end

# ╔═╡ a019634c-4762-4569-89d8-e03c3e38617e
begin
	Plots.plot(t_div, sol.(t_div),label="Solución real",lw=3, ls=:dash)
	Plots.plot!(t_div,xopt4[1,:],title="Comparación de la solución obtenida con FBS y\n la solución real", label="Solución obtenida" ,lw=2)
end

# ╔═╡ 215ae0cf-6882-4fe0-8c89-e03c9440e16e
begin
	Plots.plot(t_div, -control_real.(t_div),label="Solución adjunta real",lw=3, ls=:dash)
	Plots.plot!(t_div, lambopt4[1,:],title="Comparación de la dinámica adjunta obtenida con\n FBS y la solución adjunta real", label="Solución adjunta obtenida" ,lw=2)
end

# ╔═╡ 6d474536-40ae-4f95-a7ce-3f85f9fd613d
begin
	Plots.plot(t_div[1:Nt], control_real.(t_div[1:Nt]),label="Control óptimo real", ls=:dash, lw=3)
	Plots.plot!(t_div[1:Nt],uopt4[1,:],title="Comparación del control obtenido mediante\n FBS y control óptimo real", label="Control obtenido con NE",lw=2)
end

# ╔═╡ 7092415b-fd0c-4ced-bcc3-9374ab3798c8
md"""
# Referencias
"""

# ╔═╡ 9e9b988b-67a3-4b11-9b08-d9cd671e50a1
md"""

[1] Bryson, A. E. (1975). Applied optimal control: Optimization, Estimation and Control. CRC Press.

[2] CMU Robotic Exploration Lab. (2024, 16 febrero). Optimal Control (CMU 16-745) 2024 Lecture 10: Nonlinear Trajectory Optimization [Vídeo]. YouTube. https://www.youtube.com/watch?v=t0vaNTZIC20

[3] CMU Robotic Exploration Lab. (2024b, febrero 22). Optimal Control (CMU 16-745) 2024 Lecture 11: Differential Dynamic Programming [Vídeo]. YouTube. https://www.youtube.com/watch?v=qusvkcoHyz0

[4] Rose, Garrett Robert, "Numerical Methods for Solving Optimal Control Problems. " Master's Thesis,
University of Tennessee, 2015.
https://trace.tennessee.edu/utk_gradthes/3401 
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
ForwardDiff = "~0.10.38"
Plots = "~1.40.7"
PlutoUI = "~0.7.60"
WGLMakie = "~0.11.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "c4d251d77abaf184a4970739bba75e4821152bce"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "534820940e4359c09adc615f8bd06ca90d508ba6"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.0.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "e1371a23fd9816080c828d0ce04373857fe73d33"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.3"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7901a6117656e29fa2c74a58adb682f380922c47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.116"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "b3f2ff58735b5f024c392fde763f29b057e4b025"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.8"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.Extents]]
git-tree-sha1 = "063512a13dbe9c40d999c439268539aa552d1ae6"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.5"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "d52e255138ac21be31fa633200b65e4e71d26802"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.6"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "8e2d86e06ceb4580110d9e716be26658effc5bfd"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "da121cbdc95b065da07fbb93638367737969693f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.8+0"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "ce573eab15760315756de2c82df7406c870c7187"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.3"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "f4ee66b6b1872a4ca53303fbb51d158af1bf88d4"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.4.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "c1a9c159c3ac53aa09663d8662c7277ef3fa508d"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "dc6bed05c15523624909b3953686c5f5ffa10adc"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm_jll", "LinearAlgebra", "MacroTools", "RoundingEmulator"]
git-tree-sha1 = "ffb76d09ab0dc9f5a27edac2acec13c74a876cc6"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.21"
weakdeps = ["DiffRules", "ForwardDiff", "IntervalSets", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "021b6b64b68f6ee09fb35a1528a2b5a7f48ac00c"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.22.0"

[[deps.MakieCore]]
deps = ["ColorTypes", "GeometryBasics", "IntervalSets", "Observables"]
git-tree-sha1 = "c731269d5a2c85ffdc689127a9ba6d73e978a4b1"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.9.0"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "f45c8916e8385976e1ccd055c9874560c257ab13"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "030ea22804ef91648f29b7ad3fc15fa49d0e6e71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "f202a1ca4f6e165238d8175df63a7e26a51e04dc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "5a3a31c41e15a1e042d60f2f4942adccba05d3c9"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.0"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "3c0faa42f2bd3c6d994b06286bba2328eae34027"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.2"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "3302df2c5f9cfc692134f0d66c9a35500de9b75e"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.11.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7b5bbf1efbafb5eca466700949625e07533aff2"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+1"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "bf6bb896bd59692d1074fd69af0e5a1b64e64d5e"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.4+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "ccbb625a89ec6195856a50aa2b668a5c08712c94"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.4.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─8d88f02c-3c87-4367-af35-171c61f0a087
# ╠═3b7c3a68-51d2-48dd-ac34-3ff43e0f30fd
# ╟─cb2cef50-dcd1-11ef-18c8-035ede97281d
# ╟─adf66ca2-3f1f-4954-b2d0-1bd365e553b6
# ╟─57be2c9b-37bb-4759-b2a7-a565669b524a
# ╟─cca1cb95-cbda-4bd7-a33e-23c73b46e945
# ╠═6130a994-08c3-41f4-929c-ab9d92d72737
# ╠═0177b544-4500-4eed-9c0a-fe001e24705a
# ╠═43c83c86-f02a-48ae-9aa8-652ee4a3228c
# ╠═494776b0-ddae-4cf4-8e23-b1a888e9c716
# ╠═559c1847-f136-4730-a0ba-9ec87e936b94
# ╟─785e4e5d-ad88-4721-bee0-1b6f694df719
# ╠═6b5c18d1-5963-4288-aaf0-3e769d1bb5b1
# ╠═7bc7b335-fbbc-42ce-a349-ecf27ae9304f
# ╠═018de8d3-12b5-4d31-9a4b-25eb2bf2a9e4
# ╠═7f02b86c-1328-4505-af1c-57c676a6b75f
# ╠═2ca34ace-4a13-475b-a90d-cadd4bf730c9
# ╠═20842f67-8db2-4e64-85c4-f5b682c4c43e
# ╠═93e5bc1f-8874-46fc-85c6-9f963de80008
# ╠═830738da-3a54-454b-93a6-472b07a060e0
# ╠═6ebb5464-8c8a-4bbf-bc85-b71fbb8f186a
# ╠═305f3e99-2682-4a50-aaf8-d35b5e2ef834
# ╠═40a71cb0-c35f-4223-9483-075676f7d6d1
# ╟─2acb8668-4672-449a-a1ea-ff7f20b1246b
# ╟─ef0f10cc-13f5-4c4b-a21f-928c04761cd1
# ╠═0fb079c3-3e4f-4036-a632-ad439313711a
# ╠═81413454-4e44-4467-9a0f-b0b4a5207b06
# ╠═85ba188c-f69f-485a-8b86-4b60b07e4149
# ╠═7aa982db-0ca8-41ce-850d-e6d916a5d73c
# ╠═7689b3ec-6bdf-4d61-9317-29af64eb71db
# ╟─f70eb405-2f45-4b28-95b3-ea78b02f9b34
# ╟─5aef0348-61c6-4fd6-bfbc-c4356dcda30b
# ╟─ffd94de9-7f13-4b5c-98b5-c232d9571b32
# ╟─abae767c-d3e1-46b5-98e4-9737df833088
# ╟─27e9c6f5-0abf-4f54-aafa-524dfb9b9037
# ╟─265ecaf9-f2b4-42a3-9766-6db564c951d1
# ╟─b0f3e68a-059a-4b1a-983a-318317372238
# ╟─95aa2db6-1f1b-49e6-97de-700912d87b31
# ╟─ced75807-dec6-4d39-9a5a-690238c29272
# ╟─35c66f76-7100-40d9-85f8-cfac9796b169
# ╠═c9846053-4131-4df7-b855-25633efaa0d1
# ╟─22bff809-b3fd-499f-82ba-04341f72b4b5
# ╠═f3935d3a-07fd-4036-8e66-a7a61936df61
# ╠═d8fae788-1ab7-4a67-8bc0-227f24311aa2
# ╠═3c8814e3-2ebf-4c4d-b136-1f8827ef0e5d
# ╟─3f150fc1-93db-4602-ac10-d42b2ae754af
# ╟─1e464469-e181-4b6d-a986-4566dfdfcd0b
# ╟─c686379e-22e1-49dc-95a7-14f14113df7d
# ╟─32bf6afa-5af1-4372-9796-1ddb251ea119
# ╟─e4d073b8-4335-4417-ba5e-e751b000bb20
# ╟─ca26ab49-1c23-4f6f-b2c3-fb84fce0870e
# ╠═4604cf4b-0750-4f0d-80b4-5781df324621
# ╟─37d29234-93e6-4700-9ab5-493cbf929d06
# ╠═5ca07f66-3099-4b9e-8f2f-dd857fbb2d08
# ╠═166ed9f2-2cc5-493a-8160-76c7977e8006
# ╠═3fbcea27-1585-4bce-abde-f37b8c382ced
# ╟─e8565754-334e-4447-b064-bb340cb6f8d9
# ╟─442b2a19-9a17-44b7-b4df-91062b6bb435
# ╟─9ca1805e-c474-4139-bd7f-6ed8f44b647b
# ╟─87312b77-b624-47f4-9a2b-790d73cde8f6
# ╠═52e99cba-6a2a-43ea-ae34-c6d15ed38683
# ╠═6d933be0-8221-49c7-8776-3d5d27d0d943
# ╟─a019634c-4762-4569-89d8-e03c3e38617e
# ╟─215ae0cf-6882-4fe0-8c89-e03c9440e16e
# ╟─6d474536-40ae-4f95-a7ce-3f85f9fd613d
# ╟─7092415b-fd0c-4ced-bcc3-9374ab3798c8
# ╟─9e9b988b-67a3-4b11-9b08-d9cd671e50a1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
