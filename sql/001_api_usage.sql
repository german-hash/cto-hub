-- Registro de consumo de tokens de la API de Anthropic (solo metadatos, sin contenido).
-- Correr una vez en Supabase → SQL Editor.

create table if not exists public.api_usage (
    id                    bigserial primary key,
    created_at            timestamptz not null default now(),
    source                text not null,          -- cto_agent, news:tecnologia, granola_weekly, stock_screener...
    model                 text not null,
    input_tokens          integer not null default 0,  -- input sin caché
    output_tokens         integer not null default 0,
    cache_read_tokens     integer not null default 0,  -- leídos de caché (~10% del precio de input)
    cache_creation_tokens integer not null default 0   -- escritos en caché (~125% del precio de input)
);

create index if not exists api_usage_created_at_idx on public.api_usage (created_at desc);

-- RLS activado y sin policies: solo el service_role (el que usa el hub) puede leer/escribir.
-- La anon key no ve nada.
alter table public.api_usage enable row level security;

-- Resumen diario por fuente y modelo
create or replace view public.api_usage_daily
with (security_invoker = true) as
select
    date_trunc('day', created_at at time zone 'America/Argentina/Buenos_Aires')::date as dia,
    source,
    model,
    count(*)                    as llamadas,
    sum(input_tokens)           as input_tokens,
    sum(cache_read_tokens)      as cache_read_tokens,
    sum(cache_creation_tokens)  as cache_creation_tokens,
    sum(output_tokens)          as output_tokens,
    round(100.0 * sum(cache_read_tokens)
          / nullif(sum(input_tokens + cache_read_tokens + cache_creation_tokens), 0), 1) as pct_cache
from public.api_usage
group by 1, 2, 3
order by 1 desc, 2;
