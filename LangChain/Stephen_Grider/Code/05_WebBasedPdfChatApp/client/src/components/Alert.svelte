<script lang="ts">
	import classNames from 'classnames';

	export let onDismiss: (() => void) | null = null;
	export let type: 'error' | 'success' | 'info' | 'warning' = 'error';

	const klasses = {
		error: 'bg-red-50 border border-red-200 text-sm text-red-600 rounded-md p-4',
		success: 'bg-green-50 border border-green-200 text-sm text-green-600 rounded-md p-4',
		info: 'bg-blue-50 border border-blue-200 text-sm text-blue-600 rounded-md p-4',
		warning: 'bg-yellow-50 border border-yellow-200 text-sm text-yellow-600 rounded-md p-4'
	};
	const klass = classNames(klasses[type], 'relative');

	function handleDismiss() {
		if (onDismiss) {
			onDismiss();
		}
	}
</script>

<div class={klass} role="alert">
	<slot />

	{#if onDismiss}
		<div
			on:keydown={handleDismiss}
			on:click={handleDismiss}
			class="absolute cursor-pointer inset-y-0 right-2 flex flex-col justify-center text-red font-bold"
		>
			X
		</div>
	{/if}
</div>
