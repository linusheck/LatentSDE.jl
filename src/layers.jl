using Lux, LuxCore
export Diagonal

const NAME_TYPE = Union{Nothing, String, Symbol}

struct Diagonal{T <: NamedTuple, N <: NAME_TYPE} <:
    Lux.AbstractExplicitContainerLayer{(:layers,)}
 layers::T
 name::N
end

function Diagonal(layers...; name::NAME_TYPE=nothing)
 names = ntuple(i -> Symbol("layer_$i"), length(layers))
 return Diagonal(NamedTuple{names}(layers), name)
end

(m::Diagonal)(x, ps, st::NamedTuple) = applydiagonal(m.layers, x, ps, st)

@generated function applydiagonal(layers::NamedTuple{names},
 x::T,
 ps,
 st::NamedTuple) where {names, T}
 N = length(names)
 y_symbols = [gensym() for _ in 1:(N + 1)]
 st_symbols = [gensym() for _ in 1:N]
 getinput(i) = T <: AbstractVector ? :(x[$i:$i]) : :(x[$i:$i, :])
 calls = []
 append!(calls,
     [:(($(y_symbols[i]), $(st_symbols[i])) = Lux.apply(layers.$(names[i]),
             $(getinput(i)),
             ps.$(names[i]),
             st.$(names[i]))) for i in 1:N])
 push!(calls, :(st = NamedTuple{$names}((($(Tuple(st_symbols)...),)))))
 push!(calls, :($(y_symbols[N + 1]) = vcat($(y_symbols[1:N]...))))
 push!(calls, :(return $(y_symbols[N + 1]), st))
 return Expr(:block, calls...)
end

Base.keys(m::Diagonal) = Base.keys(getfield(m, :layers))
